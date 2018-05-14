import svgwrite
import numpy
from Robot_cls import NORTH, SOUTH, EAST, WEST, Robot
from q_learning import q_learning, get_optimal_path, make_random_policy


def render_policy (env, ofile, Q = None):
    """
    Function to render a given gridworld object. If a Q-function
    is provided, e.g. calculated using the 'extract_policy_greedily'
    function, than also the optimal policy is rendered.

    :param env: Robot_cls class object

    :param ofile: string, path to the output file.

    :param Q: array-like, shape = (nSp, nA) (default = None)
              If Q is None, than only the world is rendered,
              but not the optimal policy.
    """
    dwg = svgwrite.Drawing (ofile, profile = 'full')

    ssize = 101
    strokec = "black"
    strokew = 2.5

    # Cell colors
    termic = "gray"
    fieldc = "white"
    obstac = "black"
    dustc = "red"
    startc = "green"

    # First build up the world
    for state in range (env.nSp):
        x, y = env._state2coord (state)
        x, y = int(x), int(y)

        if env.world[y, x] == "T":
            # Terminal state
            cellc = termic
            textc = "black"
        elif env.world[y, x] == ".":
            # Normal state
            cellc = fieldc
            textc = "black"
        elif env.world[y, x] == "o":
            # Obstacle state (actually the robot will never be here)
            cellc = obstac
            textc = "white"
        elif env.world[y, x] == "x":
            cellc = dustc
            textc = "black"
        elif env.world[y, x] == "S":
            cellc = startc
            textc = "black"
        else:
            raise ValueError ("Unsupported world field: %s." % env.world[y, x])

        dwg.add (dwg.rect (insert = (x * ssize, y * ssize),
                           size = (ssize, ssize),
                           stroke = strokec,
                           stroke_width = strokew,
                           fill = cellc))

        if not Q is None:
            # If an action-value function Q is given we output the optimal policy
            # - using a greedy approach - based on Q.
            if env.world[y, x] in ["T", "o"]:
                continue

            hssize = numpy.ceil (ssize / 2.0)
            cx = x * ssize + hssize
            cy = y * ssize + hssize

            # As it can be that at a given state, two actions have the same value, i.e.
            # that they are both equally good, we display all actions having the same
            # "goodness".
            for best_action in numpy.where (Q[state] == numpy.max (Q[state]))[0]:
                if best_action == NORTH:
                    dwg.add (dwg.polyline ([(cx, cy), (cx, cy - 45)], stroke = strokec, stroke_width = strokew, fill = "none"))
                    dwg.add (dwg.polyline ([(cx + 10, cy - 35), (cx, cy - 45)], stroke = strokec, stroke_width = strokew, fill = "none"))
                    dwg.add (dwg.polyline ([(cx - 10, cy - 35), (cx, cy - 45)], stroke = strokec, stroke_width = strokew, fill = "none"))

                elif best_action == SOUTH:
                    dwg.add (dwg.polyline ([(cx, cy), (cx, cy + 45)], stroke = strokec, stroke_width = strokew, fill = "none"))
                    dwg.add (dwg.polyline ([(cx + 10, cy + 35), (cx, cy + 45)], stroke = strokec, stroke_width = strokew, fill = "none"))
                    dwg.add (dwg.polyline ([(cx - 10, cy + 35), (cx, cy + 45)], stroke = strokec, stroke_width = strokew, fill = "none"))

                elif best_action == WEST:
                    dwg.add (dwg.polyline ([(cx - 45, cy), (cx, cy)], stroke = strokec, stroke_width = strokew, fill = "none"))
                    dwg.add (dwg.polyline ([(cx - 35, cy + 10), (cx - 45, cy)], stroke = strokec, stroke_width = strokew, fill = "none"))
                    dwg.add (dwg.polyline ([(cx - 35, cy - 10), (cx - 45, cy)], stroke = strokec, stroke_width = strokew, fill = "none"))

                elif best_action == EAST:
                    dwg.add (dwg.polyline ([(cx + 45, cy), (cx, cy)], stroke = strokec, stroke_width = strokew, fill = "none"))
                    dwg.add (dwg.polyline ([(cx + 35, cy + 10), (cx + 45, cy)], stroke = strokec, stroke_width = strokew, fill = "none"))
                    dwg.add (dwg.polyline ([(cx + 35, cy - 10), (cx + 45, cy)], stroke = strokec, stroke_width = strokew, fill = "none"))
        else:
            # Output the index of each state.
            hssize = numpy.ceil (ssize / 2.0)
            cx = x * ssize + hssize
            cy = y * ssize + hssize

            dwg.add (dwg.text (str (state), insert = (cx, cy), fill = textc))

    dwg.save()

if __name__ == "__main__":
    discount_factor = 0.9
    robot = Robot(Robot._create_example_world("obstacles")) # Check also: 4x4_world, 6x4_world_2
    Q = q_learning(robot, make_random_policy(robot), robot.S_start[0], discount_factor)
    render_policy(robot, "./robo.svg", Q)
    print(Q)
    path = get_optimal_path(robot, Q)
    print(list(map(robot._state2coord, path)))