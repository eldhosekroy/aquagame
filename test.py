"""
AR Aquarium Fish Game - Follow Finger Only
Performance-optimized for Python (no GPU required).

Main performance changes:
 - Camera set to 640x360
 - Cached gradient (vectorized) (kept)
 - Adaptive Mediapipe frame-skip (less frequent hand inference when FPS drops)
 - Reduced particle counts for eat / level-up
 - Particle cap (drop oldest particles when too many)
 - Only draw Mediapipe landmarks when debug=True

Requires:
 pip install opencv-python mediapipe numpy
 Optional for sound: pip install pygame

Author: ChatGPT (adapted & optimized)
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
import collections

# Optional sound support
try:
    import pygame
    pygame.mixer.init()
    SOUND_ENABLED = True
except Exception:
    SOUND_ENABLED = False

def play_sound(name):
    if not SOUND_ENABLED:
        return
    mapping = {"eat": None, "level": None, "drop": None}
    path = mapping.get(name)
    if path:
        try:
            s = pygame.mixer.Sound(path)
            s.play()
        except Exception:
            pass

# ----------------- Mediapipe Hand Setup -----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# ----------------- Utilities -----------------
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def clamp(x, a, b):
    return max(a, min(b, x))

# ----------------- Particle System -----------------
class Particle:
    def __init__(self, pos, vel, life, color, radius=3):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.life = life
        self.max_life = life
        self.color = color
        self.radius = radius

    def update(self, dt):
        # mild gravity for splash
        self.vel += np.array((0, 60.0)) * dt
        self.pos += self.vel * dt
        self.life -= dt

    def draw(self, frame):
        if self.life <= 0:
            return
        alpha = max(0.0, self.life / self.max_life)
        col = tuple(int(c * alpha) for c in self.color)
        cv2.circle(frame, (int(self.pos[0]), int(self.pos[1])), max(1, int(self.radius * (alpha))), col, -1)

# ----------------- Food -----------------
class Food:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.radius = 8
        self.spawn_time = time.time()
        self.sink_speed = random.uniform(20, 80)  # px/sec

    def update(self, dt, h):
        # sink slowly until bottom of frame
        if self.pos[1] < h - 10:
            self.pos[1] += self.sink_speed * dt

    def draw(self, frame):
        x, y = int(self.pos[0]), int(self.pos[1])
        cv2.circle(frame, (x, y), self.radius, (0, 190, 0), -1)
        cv2.circle(frame, (x, y), int(self.radius*1.6), (0, 120, 0), 1)

# ----------------- Fish -----------------
class Fish:
    def __init__(self, pos, level=1):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array((0.0, 0.0))
        self.max_speed = 210.0  # px/sec
        self.max_force = 200.0
        self.radius = 30
        self.level = level
        self.xp = 0
        self.xp_to_next = 3 + (level - 1) * 3
        self.color = self.get_color_for_level(level)
        self.tail_phase = 0.0

    def get_color_for_level(self, level):
        palette = {1: (200, 90, 20), 2: (200, 160, 20), 3: (120, 200, 255),
                   4: (180, 60, 200), 5: (40, 220, 140)}
        return palette.get(level, (220, 180, 100))

    def update(self, dt, target_pos=None, foods=None, w=None, h=None):
        if target_pos is None:
            # No finger on screen: stop moving immediately
            self.vel[:] = 0.0
            self.tail_phase += dt * 0.2
            return

        desired = np.array(target_pos) - self.pos
        desired_norm = np.linalg.norm(desired) + 1e-6
        speed = clamp(self.max_speed * (desired_norm / 300.0), 40.0, self.max_speed)
        desired = (desired / desired_norm) * speed
        steer = desired - self.vel

        force_norm = np.linalg.norm(steer)
        if force_norm > self.max_force:
            steer = steer / force_norm * self.max_force

        self.vel += steer * dt
        vel_norm = np.linalg.norm(self.vel)
        if vel_norm > self.max_speed:
            self.vel = (self.vel / vel_norm) * self.max_speed

        self.pos += self.vel * dt
        if w:
            self.pos[0] = clamp(self.pos[0], 20, w - 20)
        if h:
            self.pos[1] = clamp(self.pos[1], 20, h - 20)
        self.tail_phase += dt * (1 + vel_norm / 100.0)

    def draw(self, frame):
        x, y = int(self.pos[0]), int(self.pos[1])
        vx, vy = self.vel[0], self.vel[1]
        angle = math.degrees(math.atan2(vy, vx)) if (abs(vx)+abs(vy))>1e-6 else 0
        body_length = int(self.radius * (1.6 + 0.1 * self.level))
        body_height = int(self.radius * (0.9 - 0.05 * self.level))

        cv2.ellipse(frame, (x, y), (body_length, body_height), angle, 0, 360, self.color, -1)
        cv2.ellipse(frame, (x, y), (int(body_length*0.6), int(body_height*0.6)), angle, 0, 360, (255, 220, 200), 1)

        tail_len = int(self.radius * (0.9 + 0.15 * math.sin(self.tail_phase * 6)))
        rad = math.radians(angle)
        back_x = int(x - math.cos(rad) * (body_length - 10))
        back_y = int(y - math.sin(rad) * (body_length - 10))
        left_angle = math.radians(angle + 110)
        right_angle = math.radians(angle - 110)
        p1 = (int(back_x + math.cos(left_angle) * tail_len), int(back_y + math.sin(left_angle) * tail_len))
        p2 = (int(back_x + math.cos(right_angle) * tail_len), int(back_y + math.sin(right_angle) * tail_len))
        cv2.fillConvexPoly(frame, np.array([ (back_x, back_y), p1, p2 ], dtype=np.int32), self.color)

        eye_offset_x = int(math.cos(rad) * (body_length * 0.4))
        eye_offset_y = int(math.sin(rad) * (body_length * 0.4))
        eye_pos = (x + eye_offset_x, y + eye_offset_y)
        cv2.circle(frame, eye_pos, int(body_height*0.18), (255,255,255), -1)
        cv2.circle(frame, (eye_pos[0]+2, eye_pos[1]), int(body_height*0.08), (30,30,30), -1)

        if self.level >= 3:
            cv2.circle(frame, (x, y), int(body_height*1.1), (200,200,255), 1)

# ----------------- Game Manager -----------------
class Game:
    def __init__(self):
        self.foods = []
        # use deque for efficient pops from left when cap exceeded
        self.particles = collections.deque()
        self.max_particles = 120  # hard cap on particle count
        self.fish = Fish((400, 240), level=1)
        self.score = 0
        self.last_food_time = 0.0
        self.min_spawn = 2.0
        self.max_spawn = 6.0
        self.next_spawn_time = time.time() + random.uniform(self.min_spawn, self.max_spawn)
        self.w = 1280
        self.h = 720
        self.debug = False

        # Gradient cache (vectorized)
        self._gradient_cache = None
        self._gradient_size = (0, 0)

    def spawn_food(self, pos):
        f = Food(pos)
        self.foods.append(f)
        self.last_food_time = time.time()
        self.next_spawn_time = time.time() + random.uniform(self.min_spawn, self.max_spawn)
        play_sound("drop")

    def spawn_food_random(self):
        rx = random.uniform(80, self.w - 80)
        ry = random.uniform(40, 120)
        self.spawn_food((rx, ry))

    def update(self, dt, hand_pos=None):
        self.fish.update(dt, target_pos=hand_pos, foods=None, w=self.w, h=self.h)

        # update foods
        for f in self.foods[:]:
            f.update(dt, self.h)
            if distance(self.fish.pos, f.pos) < (self.fish.radius * 0.9 + f.radius):
                self._fish_eat(f)
                try:
                    self.foods.remove(f)
                except ValueError:
                    pass

        # update particles
        # iterate conservative copy to avoid heavy operations
        for p in list(self.particles):
            p.update(dt)
        # remove dead particles
        while self.particles and self.particles[0].life <= 0:
            self.particles.popleft()

        # random spawn
        if time.time() >= self.next_spawn_time:
            self.spawn_food_random()

        # safety spawn if none for long
        if len(self.foods) == 0 and (time.time() - self.last_food_time) > self.max_spawn:
            self.spawn_food_random()

    def _add_particle(self, p):
        # ensure cap
        self.particles.append(p)
        if len(self.particles) > self.max_particles:
            # drop oldest to keep CPU bounded
            try:
                self.particles.popleft()
            except Exception:
                pass

    def _fish_eat(self, food):
        # reduced particle count (was 18)
        for _ in range(8):
            ang = random.uniform(0, 2*math.pi)
            speed = random.uniform(40, 140)
            vel = (math.cos(ang) * speed, math.sin(ang) * speed)
            p = Particle(food.pos.copy(), vel, life=random.uniform(0.4, 0.9),
                         color=(0, 170 + random.randint(0,80), 255), radius=random.uniform(2,4))
            self._add_particle(p)
        self.score += 1
        self.fish.xp += 1
        play_sound("eat")
        if self.fish.xp >= self.fish.xp_to_next:
            self._level_up_fish()

    def _level_up_fish(self):
        self.fish.level += 1
        self.fish.xp = 0
        self.fish.xp_to_next = 3 + (self.fish.level - 1) * 3
        self.fish.color = self.fish.get_color_for_level(self.fish.level)
        # reduced level-up particles (was 40)
        for _ in range(12):
            ang = random.uniform(0, 2*math.pi)
            speed = random.uniform(80, 260)
            vel = (math.cos(ang)*speed, math.sin(ang)*speed)
            p = Particle(self.fish.pos.copy(), vel, life=random.uniform(0.5,1.2), color=(255,220,150), radius=random.uniform(3,5))
            self._add_particle(p)
        self.fish.max_speed *= 1.06
        play_sound("level")

    def _create_gradient_for_size(self, h, w):
        t = np.linspace(0.0, 1.0, h).astype(np.float32)
        blue_vals = (60 + 40 * (1.0 - t)).astype(np.uint8)
        grad = np.empty((h, w, 3), dtype=np.uint8)
        grad[:, :, 0] = np.repeat(blue_vals[:, None], w, axis=1)
        grad[:, :, 1] = 140
        grad[:, :, 2] = 200
        return grad

    def draw(self, frame):
        h, w = frame.shape[:2]
        if self._gradient_cache is None or self._gradient_size != (h, w):
            try:
                self._gradient_cache = self._create_gradient_for_size(h, w)
                self._gradient_size = (h, w)
            except Exception:
                self._gradient_cache = None
                self._gradient_size = (h, w)

        if self._gradient_cache is not None:
            blended = (frame.astype(np.float32) * 0.9 + self._gradient_cache.astype(np.float32) * 0.1).astype(np.uint8)
            frame[:] = blended

        for f in list(self.foods):
            f.draw(frame)

        self.fish.draw(frame)

        for p in list(self.particles):
            p.draw(frame)

        cv2.rectangle(frame, (0, h-40), (w, h), (150, 120, 60), -1)
        for i in range(18):  # slightly fewer decorations
            px = int((i * 73) % w)
            py = h - 20 + int(5*math.sin(i + time.time()))
            cv2.circle(frame, (px, py), 3, (120,100,70), -1)

        # HUD
        cv2.putText(frame, f"Score: {self.score}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(frame, f"Fish Level: {self.fish.level}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        bar_x, bar_y = 20, 80
        bar_w = 220
        bar_h = 14
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60,60,60), -1)
        xp_frac = self.fish.xp / max(1, self.fish.xp_to_next)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * xp_frac), bar_y + bar_h), (0, 200, 200), -1)
        cv2.putText(frame, f"XP {self.fish.xp}/{self.fish.xp_to_next}", (bar_x + bar_w + 12, bar_y + bar_h -2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1)

        if self.debug:
            cv2.putText(frame, "Debug: ON (press 'c' to toggle)", (20, self.h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

# ----------------- Main loop with adaptive Mediapipe -----------------
def main():
    # prefer lower resolution for performance
    source = "https://10.140.217.200:8080/video"
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

    # set lower resolution (may be ignored by some IP streams, but helps local webcams)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    game = Game()
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Camera read failed")
        return
    h0, w0 = frame.shape[:2]
    game.w = w0
    game.h = h0

    prev_time = time.time()
    show_fps = True

    # adaptive mediapipe parameters
    detection_skip = 1        # process every N frames (1 = every frame)
    detection_skip_min = 1
    detection_skip_max = 6
    fps_smooth = 30.0
    alpha_fps = 0.85
    smoothed_fps = 30.0
    frame_counter = 0
    last_hand_pos = None
    last_hand_seen_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        now = time.time()
        dt = now - prev_time
        prev_time = now
        frame_counter += 1

        # compute smoothed fps
        if dt > 0:
            inst_fps = 1.0 / dt
            smoothed_fps = alpha_fps * smoothed_fps + (1 - alpha_fps) * inst_fps

        # adaptive detection skip: increase skip if fps low, decrease if fps high
        # tuning: keep skip small when > 25 fps
        if smoothed_fps < 18 and detection_skip < detection_skip_max:
            detection_skip += 1
        elif smoothed_fps > 30 and detection_skip > detection_skip_min:
            detection_skip -= 1

        hand_pos = None

        # run Mediapipe only every detection_skip frames
        run_detection_this_frame = (frame_counter % detection_skip) == 0

        if run_detection_this_frame:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                hand_pos = (ix, iy)
                last_hand_pos = hand_pos
                last_hand_seen_time = now
                if game.debug:
                    mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                # draw tiny cross only in debug; otherwise skip heavy landmark draws
                if game.debug:
                    cv2.circle(frame, (ix, iy), 6, (255,255,255), -1)
            else:
                # keep last known hand for a short grace period
                if (now - last_hand_seen_time) < 0.25:
                    hand_pos = last_hand_pos
                else:
                    hand_pos = None
        else:
            # reuse last known hand position for smoother motion when skipping detection
            if (now - last_hand_seen_time) < 0.25:
                hand_pos = last_hand_pos
            else:
                hand_pos = None

        # update game
        game.update(dt, hand_pos=hand_pos)

        # draw scene
        game.draw(frame)

        # show performance overlays
        if show_fps:
            fps = int(smoothed_fps)
            cv2.putText(frame, f"FPS: {fps} (skip={detection_skip})", (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2)

        if game.debug and hand_pos is not None:
            cv2.line(frame, (hand_pos[0]-10, hand_pos[1]), (hand_pos[0]+10, hand_pos[1]), (200,200,200), 1)
            cv2.line(frame, (hand_pos[0], hand_pos[1]-10), (hand_pos[0], hand_pos[1]+10), (200,200,200), 1)

        cv2.imshow("AR Aquarium - Guide the fish to random food (fish follows finger only)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord('c'):  # toggle debug
            game.debug = not game.debug
        if key == ord('p'):  # toggle fps display
            show_fps = not show_fps
        if key == ord('r'):  # reduce particle cap
            game.max_particles = max(20, game.max_particles - 20)
        if key == ord('R'):  # increase particle cap
            game.max_particles += 20

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

