#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    MAIN

"""

import math
import os
import sys

import neat
import pygame

# Uncomment the line below to see the program run.
# os.environ["SDL_VIDEODRIVER"] = "dummy"  # Resolves pipeline error

TRACK_ID = 3  # Select a track
PATH = "Users/samuel.hernandez/FormulaAI"

tracks = {
    0: "AutonomoHermanosRodriguez",
    1: "CircuitOfTheAmericas",
    2: "Monaco",
    3: "Monza",
}

start_pos = {
    0: (300, 920),
    1: (270, 930),
    2: (270, 570),
    3: (570, 790),
}

CURR_TRACK = tracks[TRACK_ID]

pygame.font.init()  # you have to call this at the start, if you want to use this module.
font = pygame.font.SysFont("Arial", 30)
pygame.display.set_caption("Formula AI")
TRACK = pygame.image.load(f"/{PATH}/assets/Monza_.png")


WIDTH = TRACK.get_width()
HEIGHT = TRACK.get_height()
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))

FPS = 28
clock = pygame.time.Clock()


class Car(pygame.sprite.Sprite):
    """
    "Car"

    """

    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(
            "/Users/samuel.hernandez/FormulaAI/assets/ferrari641.png"
        )
        self.image = self.original_image
        self.rect = self.image.get_rect(center=start_pos[TRACK_ID])
        self.vel_vector = pygame.math.Vector2(0.6, 0)
        self.angle = 0
        self.corner_vel = 9
        self.direction = 0
        self.on_track = True
        self.sensors = []
        self.laps = 0
        self.has_crossed_line  = False
        self.first_lap = True

    def update(self):
        """
        updates the car's heading based on the radar inputs

        """
        self.sensors.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-135, -25, 0, 25, 135):
            self.radar(radar_angle)
        self.collision()
        self.data()
        self.check_lap()

    def check_lap(self):
            start_x, start_y = start_pos[TRACK_ID]
            car_x, car_y = self.rect.center
            distance_to_start = math.sqrt((start_x - car_x) ** 2 + (start_y - car_y) ** 2)
            if distance_to_start < 30:
                if not self.has_crossed_line:
                    if not self.first_lap:
                        self.laps += 1
                    else:
                        self.first_lap = False
                    self.has_crossed_line = True
            else:
                self.has_crossed_line = False

    def drive(self):
        """
        keeps the car straight

        """
        self.rect.center += self.vel_vector * 8

    def collision(self):
        """
        In the event of a crash, this method simulates the collision

        """
        length = 10
        collision_point_right = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length),
        ]
        collision_point_left = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length),
        ]

        # Die on Collision
        if SCREEN.get_at(collision_point_right) == pygame.Color(
            0, 108, 12, 255
        ) or SCREEN.get_at(collision_point_left) == pygame.Color(0, 108, 12, 255):
            self.on_track = False

        # Draw Collision Points
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 2)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 2)

    def rotate(self):
        """
        rotates car
        """
        if self.direction == 1:
            self.angle -= self.corner_vel
            self.vel_vector.rotate_ip(self.corner_vel)
        if self.direction == -1:
            self.angle += self.corner_vel
            self.vel_vector.rotate_ip(-self.corner_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        """
        this method simulates the distance from the car to the edge of the track
        """
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while SCREEN.get_at((x, y)) != pygame.Color(0, 108, 12, 255) and length < 60:
            length += 1
            x = int(
                self.rect.center[0]
                + math.cos(math.radians(self.angle + radar_angle)) * length
            )
            y = int(
                self.rect.center[1]
                - math.sin(math.radians(self.angle + radar_angle)) * length
            )

        # Draw Radar
        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)

        dist = int(
            math.sqrt(
                math.pow(self.rect.center[0] - x, 2)
                + math.pow(self.rect.center[1] - y, 2)
            )
        )

        self.sensors.append([radar_angle, dist])

    def data(self):
        """
        stores the radar data

        """
        input_ = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.sensors):
            input_[i] = int(radar[1])
        return input_


def remove(index):
    """
    In the event of a fatal collision, this method removes the car off the track

    """
    cars.pop(index)
    ge.pop(index)
    nets.pop(index)


def eval_genomes(genomes, config):
    """
    evaluates which genome is best fit for the track

    """
    global cars, ge, nets

    cars = []
    ge = []
    nets = []
    reward_laps = 2  # set the laps for reward

    for genome_id, genome in genomes:
        cars.append(pygame.sprite.GroupSingle(Car()))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    run = True
    while run:
        clock.tick(FPS)
        text_track = f"track: {CURR_TRACK}"
        text_genome_id = f"genome id: {str(genome_id)}"
        text_laps = f"laps: {str(cars[0].sprite.laps)}" if cars else "laps: 0"
        text_surface_name = font.render(text_track, False, (255, 255, 255))
        text_surface_genome = font.render(text_genome_id, False, (255, 255, 255))
        text_surface_laps = font.render(text_laps, False, (255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))
        SCREEN.blit(text_surface_name, (0, 0))
        SCREEN.blit(text_surface_genome, (0, 30))
        SCREEN.blit(text_surface_laps, (0, 60))

        if not cars:
            break

        for i, car in enumerate(cars):
            ge[i].fitness += 1
            if not car.sprite.on_track:
                remove(i)

            if car.sprite.laps == reward_laps:
                ge[i].fitness += 3
                print(f"genome_id:{genome_id} won!\n")
                remove(i)


        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.data())
            if output[0] > 0.7:
                car.sprite.direction = 1
            if output[1] > 0.7:
                car.sprite.direction = -1
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction = 0

        # Update
        for car in cars:
            car.draw(SCREEN)
            car.update()
        pygame.display.update()


def run(config_path):
    """
    run the car

    """
    # Setup NEAT Neural Network
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    global pop
    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pop.run(eval_genomes, 500)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)
