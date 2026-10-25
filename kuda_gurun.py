import cv2
import numpy as np
import random
import json
import os
import time

# ------------------------------------------------------------------ KONFIG
W, H = 520, 720
HORSE_Y = H - 140
HX_MIN = 50
HORSE_HIT_W, HORSE_HIT_H = 38, 100
HORSE_SCALE = 2.5
SPRITE_ANCHOR = (31.5, 33.0)
START_LIVES = 3
PX_PER_M = 40.0
BASE_SPEED, MAX_SPEED, ACCEL = 260.0, 560.0, 1.2
HAND_SPEED = 380.0
MIN_ROW_GAP = 300.0
START_ROW_GAP = 520.0
ROW_DENSITY = 0.8
CACTUS_CLEARANCE = 26
DANGER_SPAN = 210.0

CACTUS_CROPS = [
    (0, 0, 16, 16, 3.0, 3), (0, 16, 16, 16, 3.0, 3), (17, 5, 14, 11, 3.2, 3), (17, 21, 14, 12, 3.2, 3),
    (16, 33, 16, 15, 3.0, 2), (16, 48, 16, 16, 3.0, 2), (0, 34, 16, 14, 3.2, 2), (0, 50, 16, 14, 3.2, 2),
    (36, 0, 28, 32, 2.2, 2), (32, 33, 29, 31, 2.2, 2),
]
INVULN_TIME = 1.5

CAM_W, CAM_H = 640, 480
MIN_HAND_AREA = 3000
MARGIN = 0.15
CONFIG_FILE = "hsv_config_ungu.json"

SAND = (145, 201, 237)


# ------------------------------------------------------------------ FILTER
class OneEuro:

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t is None:
            self.t, self.x, self.dx = t, x.copy(), np.zeros_like(x)
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


# ------------------------------------------------------------------ TRACKER
class HandTracker:
    def __init__(self):
        # default hasil ukur foto sarung tangan: H~126-129, S~100-190, V~70-230
        self.lo = [118, 70, 35]
        self.hi = [142, 255, 255]
        if os.path.exists(CONFIG_FILE):
            try:
                d = json.load(open(CONFIG_FILE))
                self.lo, self.hi = d["lo"], d["hi"]
            except Exception:
                pass
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        self.acc = None
        self.present = False
        self.pos = (0.5, 0.5)
        self.box = None
        self.mask = np.zeros((CAM_H, CAM_W), np.uint8)
        self.last_px = None
        self.lost = 0
        self._setup_trackbars()

    def _setup_trackbars(self):
        cv2.namedWindow("HSV")
        cv2.resizeWindow("HSV", 420, 250)
        names = [("H low", self.lo[0], 179), ("H high", self.hi[0], 179),
                 ("S low", self.lo[1], 255), ("S high", self.hi[1], 255),
                 ("V low", self.lo[2], 255), ("V high", self.hi[2], 255)]
        names[1] = ("H high", self.hi[0], 179)
        for n, v, mx in names:
            cv2.createTrackbar(n, "HSV", v, mx, lambda x: None)

    def _read_trackbars(self):
        g = lambda n: cv2.getTrackbarPos(n, "HSV")
        self.lo = [g("H low"), g("S low"), g("V low")]
        self.hi = [g("H high"), g("S high"), g("V high")]

    def save(self):
        json.dump({"lo": self.lo, "hi": self.hi}, open(CONFIG_FILE, "w"))
        print("Kalibrasi tersimpan:", self.lo, self.hi)

    def calibrate(self, frame):
        """Ambil sampel warna sarung tangan dari kotak di tengah kamera."""
        h, w = frame.shape[:2]
        x0, y0 = w // 2 - 50, h // 2 - 50
        blur = cv2.GaussianBlur(frame, (7, 7), 0)
        roi = cv2.cvtColor(blur[y0:y0 + 100, x0:x0 + 100], cv2.COLOR_BGR2HSV).reshape(-1, 3)
        lo = np.percentile(roi, 2, axis=0)
        hi = np.percentile(roi, 98, axis=0)
        new_lo = [max(0, int(lo[0]) - 10), max(50, int(lo[1]) - 50), max(25, int(lo[2]) - 90)]
        new_hi = [min(179, int(hi[0]) + 10), min(255, int(hi[1]) + 80), 255]
        for n, v in zip(["H low", "S low", "V low"], new_lo):
            cv2.setTrackbarPos(n, "HSV", v)
        for n, v in zip(["H high", "S high", "V high"], new_hi):
            cv2.setTrackbarPos(n, "HSV", v)
        print("Auto-kalibrasi:", new_lo, new_hi)

    def process(self, frame):
        self._read_trackbars()
        # 1) HSV + threshold warna ungu
        blur = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.lo), np.array(self.hi))

        # 2) morfologi + rata-rata antar frame (anti kedip)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        if self.acc is None or self.acc.shape != mask.shape:
            self.acc = mask.astype(np.float32)
        cv2.accumulateWeighted(mask.astype(np.float32), self.acc, 0.4)
        mask = (self.acc > 110).astype(np.uint8) * 255

        # 3) isi lubang, buang serpihan kecil
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        for k in cnts:
            if cv2.contourArea(k) > 300:
                cv2.drawContours(filled, [k], -1, 255, -1)
        self.mask = filled

        # 4) gabungkan potongan tangan jadi 1 blob, pilih blob (ikuti posisi sebelumnya)
        heavy = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, self.big_kernel)
        cnts, _ = cv2.findContours(heavy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = [k for k in cnts if cv2.contourArea(k) > MIN_HAND_AREA]
        c = None
        if cands:
            if self.last_px is not None:
                def dist_prev(k):
                    m = cv2.moments(k)
                    return np.hypot(m["m10"] / m["m00"] - self.last_px[0],
                                    m["m01"] / m["m00"] - self.last_px[1])
                c = min(cands, key=dist_prev)
            else:
                c = max(cands, key=cv2.contourArea)

        self.present = False
        if c is not None:
            # 5) kotak (bounding box) dari piksel ungu -> posisi tengah kotak
            blob = np.zeros_like(mask)
            cv2.drawContours(blob, [c], -1, 255, -1)
            pts = cv2.findNonZero(cv2.bitwise_and(filled, blob))
            bx, by, bw, bh = cv2.boundingRect(pts)
            self.box = (bx, by, bw, bh)
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            self.pos = (cx / frame.shape[1], cy / frame.shape[0])
            self.last_px, self.lost, self.present = (cx, cy), 0, True
        if not self.present:
            self.lost += 1
            if self.lost > 10:
                self.last_px = None
        return self.present

    def debug_view(self, frame):
        out = frame.copy()
        hh, ww = out.shape[:2]
        if self.present:
            bx, by, bw, bh = self.box
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
            cv2.circle(out, (int(self.pos[0] * ww), int(self.pos[1] * hh)), 8, (0, 0, 255), -1)
            cv2.putText(out, f"Terdeteksi  x={self.pos[0]:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(out, "Sarung tangan tidak terdeteksi", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # zona gerak yang dipetakan ke layar game
        cv2.line(out, (int(MARGIN * ww), 0), (int(MARGIN * ww), hh), (255, 255, 255), 1)
        cv2.line(out, (int((1 - MARGIN) * ww), 0), (int((1 - MARGIN) * ww), hh), (255, 255, 255), 1)
        cv2.rectangle(out, (ww // 2 - 50, hh // 2 - 50), (ww // 2 + 50, hh // 2 + 50), (255, 255, 0), 1)
        cv2.putText(out, "K = kalibrasi warna (telapak di kotak kuning)", (10, hh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        small = cv2.resize(cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR), (160, 120))
        out[-120:, -160:] = small
        return out


# ------------------------------------------------------------------ GAMBAR
def load_horse_sprite():
    """Muat assets/horse/north.png (folder assets harus di samping file .py ini)."""
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    for folder in (base, os.getcwd()):
        path = os.path.join(folder, "assets", "horse", "north.png")
        if os.path.exists(path):
            im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if im is not None and im.ndim == 3 and im.shape[2] == 4:
                size = int(im.shape[1] * HORSE_SCALE)
                return cv2.resize(im, (size, size), interpolation=cv2.INTER_NEAREST)
    print("PERINGATAN: assets/horse/north.png tidak ditemukan -> pakai kuda gambar bawaan")
    return None


def load_cactus_sprites():
    """Potong assets/cactus/pixel_cactus.png jadi sprite-sprite terpisah (dipangkas ke tepi gambar)."""
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    for folder in (base, os.getcwd()):
        path = os.path.join(folder, "assets", "cactus", "pixel_cactus.png")
        if not os.path.exists(path):
            continue
        sheet = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if sheet is None or sheet.ndim != 3 or sheet.shape[2] != 4:
            continue
        out = []
        for (x, y, w, h, scale, weight) in CACTUS_CROPS:
            crop = sheet[y:y + h, x:x + w]
            ys, xs = np.where(crop[..., 3] > 0)
            if len(xs) == 0:
                continue
            crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            out.append({"img": crop, "w": crop.shape[1], "h": crop.shape[0], "weight": weight})
        if out:
            return out
    print("PERINGATAN: assets/cactus/pixel_cactus.png tidak ditemukan -> pakai kaktus gambar bawaan")
    return None


def overlay_rgba(img, sprite, cx, cy, anchor):
    """Tempel sprite RGBA dengan titik anchor (px pada sprite) di (cx, cy); aman di tepi layar."""
    sh, sw = sprite.shape[:2]
    x0 = int(round(cx - anchor[0]))
    y0 = int(round(cy - anchor[1]))
    ix0, iy0 = max(x0, 0), max(y0, 0)
    ix1, iy1 = min(x0 + sw, img.shape[1]), min(y0 + sh, img.shape[0])
    if ix0 >= ix1 or iy0 >= iy1:
        return
    sp = sprite[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
    a = sp[..., 3:4].astype(np.float32) / 255.0
    roi = img[iy0:iy1, ix0:ix1].astype(np.float32)
    img[iy0:iy1, ix0:ix1] = (sp[..., :3] * a + roi * (1 - a)).astype(np.uint8)


def draw_heart(img, x, y, s, col):
    cv2.circle(img, (x - s // 2, y), s // 2 + 1, col, -1)
    cv2.circle(img, (x + s // 2, y), s // 2 + 1, col, -1)
    pts = np.array([[x - s - 1, y + 2], [x + s + 1, y + 2], [x, y + int(s * 1.6)]], np.int32)
    cv2.fillPoly(img, [pts], col)


def text(img, s, pos, scale=0.8, col=(255, 255, 255), th=2):
    cv2.putText(img, s, (pos[0] + 2, pos[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), th + 2)
    cv2.putText(img, s, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th)


def draw_cactus(img, x, y, r, seed):
    x, y, r = int(x), int(y), int(r)
    cv2.ellipse(img, (x + 6, y + 8), (r, int(r * 0.9)), 0, 0, 360, (110, 160, 195), -1)   # bayangan
    dark, light = (45, 105, 40), (80, 170, 70)
    for ang in (30, 150, 270, 90, 210, 330):                                              # lengan
        a = np.radians(ang + seed % 40)
        cv2.circle(img, (x + int(np.cos(a) * r * 0.9), y + int(np.sin(a) * r * 0.9)), int(r * 0.42), dark, -1)
    cv2.circle(img, (x, y), r, dark, -1)
    cv2.circle(img, (x, y), int(r * 0.78), light, -1)
    cv2.circle(img, (x, y), int(r * 0.3), dark, 2)
    for ang in range(0, 360, 40):                                                         # duri
        a = np.radians(ang + seed % 20)
        cv2.circle(img, (x + int(np.cos(a) * r * 0.62), y + int(np.sin(a) * r * 0.62)), 2, (235, 240, 240), -1)
    if seed % 3 == 0:                                                                     # bunga
        cv2.circle(img, (x, y), 5, (150, 90, 235), -1)


def draw_horse(img, x, y, phase):
    x, y = int(x), int(y)
    brown, dark, mane = (43, 90, 139), (30, 62, 100), (20, 30, 45)
    cv2.ellipse(img, (x + 8, y + 8), (24, 58), 0, 0, 360, (105, 150, 185), -1)           # bayangan
    sw = int(8 * np.sin(phase))
    for (lx, ly, sgn) in ((-22, -28, 1), (22, -28, -1), (-22, 28, -1), (22, 28, 1)):     # kaki
        cv2.ellipse(img, (x + lx, y + ly + sgn * sw), (6, 16), 0, 0, 360, dark, -1)
    tail = int(7 * np.sin(phase * 0.7))                                                   # ekor
    pts = np.array([[x, y + 48], [x + tail, y + 66], [x + tail * 2, y + 82]], np.int32)
    cv2.polylines(img, [pts], False, mane, 6)
    cv2.ellipse(img, (x, y), (22, 52), 0, 0, 360, brown, -1)                              # badan
    cv2.ellipse(img, (x, y + 4), (13, 16), 0, 0, 360, (40, 40, 180), -1)                  # pelana
    cv2.ellipse(img, (x, y - 52), (11, 22), 0, 0, 360, brown, -1)                         # leher + kepala
    cv2.ellipse(img, (x, y - 70), (9, 14), 0, 0, 360, dark, -1)                           # moncong
    cv2.circle(img, (x - 6, y - 60), 2, (0, 0, 0), -1)
    cv2.circle(img, (x + 6, y - 60), 2, (0, 0, 0), -1)
    cv2.ellipse(img, (x - 8, y - 42), (4, 7), 0, 0, 360, dark, -1)                        # telinga
    cv2.ellipse(img, (x + 8, y - 42), (4, 7), 0, 0, 360, dark, -1)
    cv2.line(img, (x, y - 48), (x, y - 12), mane, 4)                                      # surai


# ------------------------------------------------------------------ GAME
class Game:
    def __init__(self):
        rnd = random.Random(7)
        self.dunes = [(rnd.randint(0, W), rnd.randint(0, H + 100), rnd.randint(14, 46),
                       rnd.randint(4, 12), rnd.choice((-14, -8, 10, 16))) for _ in range(46)]
        self.sprite = load_horse_sprite()
        self.cactus_sprites = load_cactus_sprites()
        self.reset()

    def reset(self):
        self.dust = []
        self.hx = W / 2
        self.lives = START_LIVES
        self.dist = 0.0
        self.speed = BASE_SPEED
        self.cacti = []
        self.since_row = 0.0
        self.prev_safe = [(HX_MIN, W - HX_MIN)]   # rentang posisi aman baris terakhir
        self.inv = 0.0
        self.over = False
        self.flash = 0.0
        self.filt = OneEuro()
        self.best = getattr(self, "best", 0)

    @property
    def score(self):
        return int(self.dist / PX_PER_M)

    def set_target(self, nx, now):
        target = HX_MIN + float(np.clip(nx, 0, 1)) * (W - 2 * HX_MIN)
        self.hx = float(np.clip(self.filt(np.array([target]), now)[0], HX_MIN, W - HX_MIN))

    # ---- spawn kaktus
    # 1) kaktus dalam 1 baris TIDAK menumpuk (ada jarak kosong CACTUS_CLEARANCE)
    # 2) selalu ada celah yang cukup lebar untuk kuda
    # 3) celah itu bisa dijangkau dari celah baris sebelumnya dengan kecepatan tangan yang wajar
    def pick_cactus(self):
        if self.cactus_sprites is None:
            r = random.randint(22, 34)
            return {"sprite": None, "w": 2 * r, "h": 2 * r, "r": r, "seed": random.randint(0, 999)}
        # makin jauh, kaktus besar makin sering muncul
        big_boost = 1.0 + self.score / 150.0
        weights = [sp["weight"] * (big_boost if sp["w"] > 55 else 1.0) for sp in self.cactus_sprites]
        sp = random.choices(self.cactus_sprites, weights=weights)[0]
        return {"sprite": sp["img"], "w": sp["w"], "h": sp["h"], "r": sp["w"] / 2, "seed": random.randint(0, 999)}

    def spawn_row(self, row_gap_px):
        reach = HAND_SPEED * max(row_gap_px - DANGER_SPAN, 40.0) / max(self.speed, 1.0) * 0.8
        n_max = min(4, 1 + self.score // 90 + 1)
        n_min = 1 if self.score < 60 else 2        # makin jauh, baris makin ramai
        for n in range(random.randint(min(n_min, n_max), n_max), 0, -1):
            for _ in range(80):
                kinds = [self.pick_cactus() for _ in range(n)]
                xs = sorted(random.uniform(k["w"] / 2 + 8, W - k["w"] / 2 - 8) for k in random.sample(kinds, n))
                items = sorted(zip(xs, kinds), key=lambda t: t[0])
                # tidak boleh menumpuk
                if any(items[i + 1][0] - items[i][0] < (items[i][1]["w"] + items[i + 1][1]["w"]) / 2 + CACTUS_CLEARANCE
                       for i in range(len(items) - 1)):
                    continue
                # celah aman untuk kuda (pakai lebar hitbox kaktus)
                edges = [(0, 0)] + [(x, k["w"] * 0.35) for x, k in items] + [(W, 0)]
                safe = []
                for i in range(len(edges) - 1):
                    lo = edges[i][0] + edges[i][1]
                    hi = edges[i + 1][0] - edges[i + 1][1]
                    if hi - lo >= HORSE_HIT_W + 80:
                        safe.append((lo + HORSE_HIT_W / 2 + 10, hi - HORSE_HIT_W / 2 - 10))
                if not safe:
                    continue
                # Rentang posisi yang BENAR-BENAR bisa dicapai kuda di baris ini = celah baris ini yang terjangkau
                # dari rentang posisi baris sebelumnya dalam waktu yang tersedia. Harus ada (dan tidak terlalu sempit),
                # sehingga selalu ada jalur yang bisa dilewati pemain yang melihat baris-baris di depan.
                nxt = []
                for (a0, a1) in self.prev_safe:
                    for (b0, b1) in safe:
                        lo, hi = max(b0, a0 - reach), min(b1, a1 + reach)
                        if lo <= hi:
                            nxt.append((lo, hi))
                if nxt and sum(hi - lo for lo, hi in nxt) >= 30:
                    for x, k in items:
                        k.update({"x": x, "y": -60.0 - random.uniform(0, 20)})
                        self.cacti.append(k)
                    self.prev_safe = nxt
                    return
        # gagal -> baris kosong. Kuda punya waktu 1 jarak baris penuh untuk bergeser, jadi rentang
        # aman sebelumnya diperlebar sebesar jarak yang bisa ditempuh (bukan di-reset penuh).
        extra = HAND_SPEED * row_gap_px / max(self.speed, 1.0) * 0.8
        self.prev_safe = [(max(HX_MIN, a0 - extra), min(W - HX_MIN, a1 + extra)) for (a0, a1) in self.prev_safe]

    def hit(self, c):
        # kotak tabrak kaktus = 70% ukuran sprite; kotak kuda = HORSE_HIT_W x HORSE_HIT_H
        return (abs(c["x"] - self.hx) < (c["w"] * 0.7 + HORSE_HIT_W) / 2 and
                abs(c["y"] - HORSE_Y) < (c["h"] * 0.7 + HORSE_HIT_H) / 2)

    def update(self, dt):
        if self.over:
            return
        self.speed = min(BASE_SPEED + self.score * ACCEL, MAX_SPEED)
        step = self.speed * dt
        self.dist += step
        self.since_row += step
        self.inv = max(0.0, self.inv - dt)
        for c in self.cacti:
            c["y"] += step
        self.cacti = [c for c in self.cacti if c["y"] < H + 60]
        # debu pasir di belakang kuda
        if random.random() < 0.5:
            self.dust.append({"x": self.hx + random.uniform(-14, 14), "y": HORSE_Y + 55.0, "r": random.uniform(4, 8), "life": 1.0})
        for d in self.dust:
            d["y"] += step * 0.9
            d["life"] -= dt * 2.2
        self.dust = [d for d in self.dust if d["life"] > 0]

        gap = max(MIN_ROW_GAP, START_ROW_GAP - self.score * ROW_DENSITY)   # makin jauh -> makin rapat
        if self.since_row >= gap:
            self.since_row = 0.0
            self.spawn_row(gap)

        if self.inv <= 0:
            for c in self.cacti:
                if self.hit(c):
                    self.cacti = [k for k in self.cacti if k is not c]   # hapus berdasarkan identitas
                    self.lives -= 1
                    self.inv = INVULN_TIME
                    self.flash = time.time()
                    if self.lives <= 0:
                        self.over = True
                        self.best = max(self.best, self.score)
                    break

    # ---- gambar
    def draw(self, waiting):
        img = np.full((H, W, 3), SAND, dtype=np.uint8)
        for (x, y0, w, h, shade) in self.dunes:
            y = int((y0 + self.dist * 0.9) % (H + 100)) - 50
            col = tuple(int(np.clip(v + shade, 0, 255)) for v in SAND)
            cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, col, -1)

        for c in self.cacti:
            if c["sprite"] is not None:
                cv2.ellipse(img, (int(c["x"]) + 7, int(c["y"] + c["h"] * 0.30)),
                            (int(c["w"] * 0.42), int(c["h"] * 0.28)), 0, 0, 360, (105, 150, 185), -1)
                overlay_rgba(img, c["sprite"], c["x"], c["y"], (c["w"] / 2, c["h"] / 2))
            else:
                draw_cactus(img, c["x"], c["y"], c["r"], c["seed"])

        for d in self.dust:
            col = tuple(int(min(255, v + 20)) for v in SAND)
            cv2.circle(img, (int(d["x"]), int(d["y"])), max(1, int(d["r"] * (1.3 - d["life"] * 0.3))), col, -1)

        blink = self.inv > 0 and int(self.inv * 10) % 2 == 0
        if not blink:
            if self.sprite is not None:
                bob = int(3 * np.sin(self.dist / 14.0))                  # hentakan lari
                cv2.ellipse(img, (int(self.hx) + 8, HORSE_Y + 12), (22, 54), 0, 0, 360, (105, 150, 185), -1)
                overlay_rgba(img, self.sprite, self.hx, HORSE_Y + bob,
                             (SPRITE_ANCHOR[0] * HORSE_SCALE, SPRITE_ANCHOR[1] * HORSE_SCALE))
            else:
                draw_horse(img, self.hx, HORSE_Y, self.dist / 18.0)

        if time.time() - self.flash < 0.2:            # kilatan merah saat terkena
            red = np.full_like(img, (0, 0, 255))
            img = cv2.addWeighted(img, 0.65, red, 0.35, 0)

        # HUD
        cv2.rectangle(img, (0, 0), (W, 62), (60, 45, 30), -1)
        text(img, f"SKOR: {self.score} m", (14, 40), 0.9)
        for i in range(START_LIVES):
            col = (60, 60, 235) if i < self.lives else (95, 95, 95)
            draw_heart(img, W - 100 + i * 34, 24, 9, col)
        text(img, f"{int(self.speed / PX_PER_M * 3.6)} km/j", (W - 130, H - 14), 0.55, (60, 45, 30), 1)

        if waiting and not self.over:
            ov = img.copy()
            cv2.rectangle(ov, (0, 0), (W, H), (0, 0, 0), -1)
            img = cv2.addWeighted(ov, 0.55, img, 0.45, 0)
            text(img, "PAUSE", (W // 2 - 70, H // 2 - 20), 1.4, (0, 255, 255), 3)
            text(img, "Angkat sarung tangan ungu", (W // 2 - 175, H // 2 + 25), 0.75)
            text(img, "ke depan kamera untuk main", (W // 2 - 180, H // 2 + 55), 0.75)

        if self.over:
            ov = img.copy()
            cv2.rectangle(ov, (0, 0), (W, H), (0, 0, 0), -1)
            img = cv2.addWeighted(ov, 0.7, img, 0.3, 0)
            text(img, "GAME OVER", (W // 2 - 140, H // 2 - 40), 1.5, (60, 60, 255), 4)
            text(img, f"Jarak / Skor: {self.score} m", (W // 2 - 150, H // 2 + 10), 0.9)
            text(img, f"Terbaik: {self.best} m", (W // 2 - 95, H // 2 + 50), 0.8, (0, 255, 255))
            text(img, "Tekan R untuk main lagi", (W // 2 - 150, H // 2 + 95), 0.7, (210, 210, 210), 1)
        return img


# ------------------------------------------------------------------ MAIN
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    tracker = HandTracker()
    game = Game()
    mouse = {"x": W / 2}
    mouse_mode = not cap.isOpened()

    def on_mouse(ev, x, y, flags, _):
        mouse["x"] = x

    cv2.namedWindow("Game")
    cv2.setMouseCallback("Game", on_mouse)
    last = time.time()
    lost_since = time.time()

    while True:
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok:
            frame = cv2.flip(frame, 1)               # mirror: tangan ke kanan = kuda ke kanan
        else:
            frame = np.zeros((CAM_H, CAM_W, 3), np.uint8)
            mouse_mode = True

        now = time.time()
        dt = min(now - last, 0.05)
        last = now

        if mouse_mode:
            present = True
            game.set_target(mouse["x"] / W, now)
            cam_view = frame
        else:
            present = tracker.process(frame)
            cam_view = tracker.debug_view(frame)
            if present:
                nx = (tracker.pos[0] - MARGIN) / (1 - 2 * MARGIN)
                game.set_target(nx, now)
                lost_since = now
            else:
                game.filt.reset()

        # pause otomatis kalau tangan hilang > 0.4 detik
        waiting = (not mouse_mode) and (now - lost_since > 0.4)
        if not waiting:
            game.update(dt)

        cv2.imshow("Game", game.draw(waiting))
        cv2.imshow("Kamera + Deteksi Tangan", cam_view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            game.reset()
        elif key == ord("s"):
            tracker.save()
        elif key == ord("k") and not mouse_mode:
            tracker.calibrate(frame)
        elif key == ord("m"):
            mouse_mode = not mouse_mode

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
