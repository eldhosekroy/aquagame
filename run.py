"""
AR Aquarium Fish Game - Follow Finger Only

Behavior changes:
 - Fish follows your finger (index-tip) ONLY.
 - If no finger detected, fish remains still (no wandering).
 - Food still spawns randomly; guide the fish to food by moving your finger.
 - Esc to quit, 'c' to toggle debug overlays.

Requires:
 pip install opencv-python mediapipe numpy
 Optional for sound: pip install pygame

Author: ChatGPT (adapted)
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time

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
    # You can replace with actual WAV files in the working directory
    mapping = {
        "eat": None,        # "eat.wav"
        "level": None,      # "levelup.wav"
        "drop": None,       # "drop.wav"
    }
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
        self.collected = False

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
    """
    Simple steering fish that now follows only the hand target_pos.
    If target_pos is None, fish remains still (no wandering).
    """
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
        palette = {
            1: (200, 90, 20),
            2: (200, 160, 20),
            3: (120, 200, 255),
            4: (180, 60, 200),
            5: (40, 220, 140),
        }
        return palette.get(level, (220, 180, 100))

    def update(self, dt, target_pos=None, foods=None, w=None, h=None):
        """
        Now: target_pos drives movement. If target_pos is None -> stop moving.
        foods argument is ignored for steering (kept for compatibility).
        """
        if target_pos is None:
            # No finger on screen: stop moving immediately
            self.vel[:] = 0.0
            # Do not update position
            # tail_phase still ticks slowly for subtle idle motion (optional)
            self.tail_phase += dt * 0.2
            return

        # steering towards the hand ONLY
        steer = np.array((0.0, 0.0))
        desired = np.array(target_pos) - self.pos
        desired_norm = np.linalg.norm(desired) + 1e-6
        # scale speed by distance so motion looks natural
        speed = clamp(self.max_speed * (desired_norm / 300.0), 40.0, self.max_speed)
        desired = (desired / desired_norm) * speed
        steer = desired - self.vel

        # limit force
        force_norm = np.linalg.norm(steer)
        if force_norm > self.max_force:
            steer = steer / force_norm * self.max_force

        # integrate
        self.vel += steer * dt
        vel_norm = np.linalg.norm(self.vel)
        if vel_norm > self.max_speed:
            self.vel = (self.vel / vel_norm) * self.max_speed

        self.pos += self.vel * dt

        # keep inside bounds
        if w:
            self.pos[0] = clamp(self.pos[0], 20, w - 20)
        if h:
            self.pos[1] = clamp(self.pos[1], 20, h - 20)

        # tail waving faster when moving
        self.tail_phase += dt * (1 + vel_norm / 100.0)

    def draw(self, frame):
        x, y = int(self.pos[0]), int(self.pos[1])
        vx, vy = self.vel[0], self.vel[1]
        angle = math.degrees(math.atan2(vy, vx)) if (abs(vx)+abs(vy))>1e-6 else 0

        body_length = int(self.radius * (1.6 + 0.1 * self.level))
        body_height = int(self.radius * (0.9 - 0.05 * self.level))

        # body
        cv2.ellipse(frame, (x, y), (body_length, body_height), angle, 0, 360, self.color, -1)
        cv2.ellipse(frame, (x, y), (int(body_length*0.6), int(body_height*0.6)), angle, 0, 360, (255, 220, 200), 1)

        # tail
        tail_len = int(self.radius * (0.9 + 0.15 * math.sin(self.tail_phase * 6)))
        rad = math.radians(angle)
        back_x = int(x - math.cos(rad) * (body_length - 10))
        back_y = int(y - math.sin(rad) * (body_length - 10))
        left_angle = math.radians(angle + 110)
        right_angle = math.radians(angle - 110)
        p1 = (int(back_x + math.cos(left_angle) * tail_len), int(back_y + math.sin(left_angle) * tail_len))
        p2 = (int(back_x + math.cos(right_angle) * tail_len), int(back_y + math.sin(right_angle) * tail_len))
        cv2.fillConvexPoly(frame, np.array([ (back_x, back_y), p1, p2 ], dtype=np.int32), self.color)

        # eye
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
        self.particles = []
        self.fish = Fish((400, 240), level=1)
        self.score = 0
        self.last_food_time = 0.0
        self.min_spawn = 2.0
        self.max_spawn = 6.0
        self.next_spawn_time = time.time() + random.uniform(self.min_spawn, self.max_spawn)
        self.w = 1280
        self.h = 720
        self.debug = True

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
        # fish now follows finger only; pass hand_pos through
        self.fish.update(dt, target_pos=hand_pos, foods=None, w=self.w, h=self.h)

        # update foods
        for f in self.foods[:]:
            f.update(dt, self.h)
            # check eat collision (only possible if you guide fish to food)
            if distance(self.fish.pos, f.pos) < (self.fish.radius * 0.9 + f.radius):
                self._fish_eat(f)
                try:
                    self.foods.remove(f)
                except ValueError:
                    pass

        # update particles
        for p in self.particles[:]:
            p.update(dt)
            if p.life <= 0:
                try:
                    self.particles.remove(p)
                except ValueError:
                    pass

        # random spawn
        if time.time() >= self.next_spawn_time:
            self.spawn_food_random()

        # safety spawn if none for long
        if len(self.foods) == 0 and (time.time() - self.last_food_time) > self.max_spawn:
            self.spawn_food_random()

    def _fish_eat(self, food):
        for _ in range(18):
            ang = random.uniform(0, 2*math.pi)
            speed = random.uniform(40, 180)
            vel = (math.cos(ang) * speed, math.sin(ang) * speed)
            p = Particle(food.pos.copy(), vel, life=random.uniform(0.5, 1.0), color=(0, 170 + random.randint(0,80), 255), radius=random.uniform(2,4))
            self.particles.append(p)
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
        for _ in range(40):
            ang = random.uniform(0, 2*math.pi)
            speed = random.uniform(80, 300)
            vel = (math.cos(ang)*speed, math.sin(ang)*speed)
            p = Particle(self.fish.pos.copy(), vel, life=random.uniform(0.6,1.6), color=(255,220,150), radius=random.uniform(3,6))
            self.particles.append(p)
        self.fish.max_speed *= 1.08
        play_sound("level")

    def draw(self, frame):
        h, w = frame.shape[:2]
        # gradient overlay
        overlay = frame.copy().astype(float)
        for i in range(h):
            t = i / h
            blue = int(60 + 40 * (1-t))
            overlay[i,:,:] = overlay[i,:,:] * (0.9) + np.array(((blue, 140, 200))) * 0.1
        frame[:] = overlay.astype(np.uint8)

        for f in list(self.foods):
            f.draw(frame)

        self.fish.draw(frame)

        for p in list(self.particles):
            p.draw(frame)

        cv2.rectangle(frame, (0, h-40), (w, h), (150, 120, 60), -1)
        for i in range(25):
            px = int((i * 73) % w)
            py = h - 20 + int(6*math.sin(i + time.time()))
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

# ----------------- Gesture detection helper -----------------
def count_extended_fingers(lm):
    tips = [8, 12, 16, 20]
    count = 0
    for t in tips:
        if lm[t].y < lm[t - 2].y:
            count += 1
    thumb_ext = lm[4].x < lm[3].x if lm[2].x < lm[3].x else lm[4].x > lm[3].x
    return count, thumb_ext

# ----------------- Main loop -----------------
def main():
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("https://10.140.217.200:8080/video")
    if not cap.isOpened():
        print("Cannot open camera")
        return

    game = Game()
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        return
    h0, w0 = frame.shape[:2]
    game.w = w0
    game.h = h0

    prev_time = time.time()
    show_fps = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        now = time.time()
        dt = now - prev_time
        prev_time = now

        hand_pos = None

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            hand_pos = (ix, iy)
            if game.debug:
                mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (ix, iy), 6, (255,255,255), -1)

        # update game (fish follows finger ONLY)
        game.update(dt, hand_pos=hand_pos)

        # draw everything
        game.draw(frame)

        if show_fps:
            fps = int(1.0 / (dt + 1e-6))
            cv2.putText(frame, f"FPS: {fps}", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2)

        if game.debug and hand_pos is not None:
            cv2.line(frame, (hand_pos[0]-10, hand_pos[1]), (hand_pos[0]+10, hand_pos[1]), (200,200,200), 1)
            cv2.line(frame, (hand_pos[0], hand_pos[1]-10), (hand_pos[0], hand_pos[1]+10), (200,200,200), 1)

        cv2.imshow("AR Aquarium - Guide the fish to random food (fish follows finger only)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('c'):
            game.debug = not game.debug
        if key == ord('p'):
            show_fps = not show_fps

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
