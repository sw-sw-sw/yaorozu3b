import time
import pymunk
import Box2D
import sys

def print_progress(engine, step, total_steps):
    percent = (step / total_steps) * 100
    sys.stdout.write(f"\r{engine} Progress: [{step}/{total_steps}] {percent:.1f}%")
    sys.stdout.flush()

def pymunk_simulation(num_bodies=100, steps=1000):
    space = pymunk.Space()
    space.gravity = (0, -981)  # pymunkはメートル単位なので、9.81 m/s^2 を 981 cm/s^2 に変換

    # 地面を作成
    segment_shape = pymunk.Segment(space.static_body, (0, 0), (800, 0), 0.0)
    segment_shape.friction = 1.0
    space.add(segment_shape)

    # 複数の物体を作成
    for _ in range(num_bodies):
        mass = 1
        radius = 25
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = 400, 300
        shape = pymunk.Circle(body, radius)
        shape.friction = 0.5
        space.add(body, shape)

    start_time = time.time()
    for step in range(steps):
        space.step(1/60)
        if step % 10 == 0:  # 10ステップごとに進捗を表示
            print_progress("Pymunk", step, steps)
    end_time = time.time()

    print()  # 進捗表示後に改行
    return end_time - start_time

def box2d_simulation(num_bodies=100, steps=1000):
    world = Box2D.b2World(gravity=(0, -9.81))

    # 地面を作成
    ground_body = world.CreateStaticBody(
        position=(0, 0),
        shapes=Box2D.b2EdgeShape(vertices=[(0, 0), (8, 0)])
    )

    # 複数の物体を作成
    for _ in range(num_bodies):
        world.CreateDynamicBody(
            position=(4, 3),
            fixtures=Box2D.b2FixtureDef(
                shape=Box2D.b2CircleShape(radius=0.25),
                density=1.0,
                friction=0.5,
            )
        )

    start_time = time.time()
    for step in range(steps):
        world.Step(1/60, 6, 2)
        if step % 10 == 0:  # 10ステップごとに進捗を表示
            print_progress("Box2D", step, steps)
    end_time = time.time()

    print()  # 進捗表示後に改行
    return end_time - start_time

if __name__ == "__main__":
    num_bodies = 3000
    steps = 100

    print(f"Running simulations with {num_bodies} bodies for {steps} steps...")

    print("\nRunning Pymunk simulation:")
    pymunk_time = pymunk_simulation(num_bodies, steps)

    print("\nRunning Box2D simulation:")
    box2d_time = box2d_simulation(num_bodies, steps)

    print(f"\nPymunk simulation time: {pymunk_time:.4f} seconds")
    print(f"Box2D simulation time: {box2d_time:.4f} seconds")

    if pymunk_time < box2d_time:
        print("Pymunk was faster in this test.")
    elif box2d_time < pymunk_time:
        print("Box2D was faster in this test.")
    else:
        print("Both engines performed equally in this test.")

    print(f"Pymunk is {box2d_time / pymunk_time:.2f}x faster than Box2D")
    print(f"Box2D is {pymunk_time / box2d_time:.2f}x faster than Pymunk")