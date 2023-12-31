import numpy as np
from mpe.coverageWord import CoverageWorld
from mpe.core import Landmark, Agent, ENERGY_RADIUS, OBSTACLE_RADIUS
from mpe.scenario import BaseScenario

obstacle = np.array([
    [[0.1, 0.1], [0.1, 0.3], [0.3, 0.3], [0.3, 0.1]],
    [[-0.1, -0.1], [-0.1, -0.3], [-0.3, -0.3], [-0.3, -0.1]]
])

observation_obstacle = obstacle.reshape(-1)

REWARD = {
    # 'collision': -1,
    # 'energy': 1,
    # 'obstacle': -1,
    # 'out_of_bound': -1,
    # 'cover': 0.1
    'collision': -30.0,
    'unconnected': -50.0,
    'cover': 75.0,
    'done': 1500.0,
    'out_of_bound': -100.0,
}

CONFIG = {
    "r_cover": 0.25,
    "r_comm": 1.0,
    "agent_size": 0.02,
    "energy": 5.0,
    "max_speed": 1.0,
    "comm_r_scale": 0.9
}

class Scenario(BaseScenario):
    def make_world(self):
        num_agents = 4
        num_energy_landmark = 20
        num_obstacles_landmark = 0

        world = CoverageWorld(num_agents = num_agents, obstacle=obstacle, comm_r_scale=CONFIG["comm_r_scale"])
        # set any world properties first
        world.dim_c = 2

        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_energy_landmark + num_obstacles_landmark)]

        # for i in range(num_energy_landmark):
        #     world.landmarks[i].color = np.array([0.25, 0.25, 0.25])
        #     world.landmarks[i].size = ENERGY_RADIUS

        # for i in range(num_obstacles_landmark):
        #     world.landmarks[i + num_energy_landmark].color = np.array([0.75, 0.75, 0.75])
        #     world.landmarks[i + num_energy_landmark].size = OBSTACLE_RADIUS
            
        for i, agent in enumerate(world.agents):
            agent.name = "agent_%d" % i
            agent.collide = False
            agent.silent = True
            agent.size = CONFIG["agent_size"]
            agent.r_cover = CONFIG["r_cover"]
            agent.r_comm = CONFIG["r_comm"]
            agent.max_speed = CONFIG["max_speed"]

        for i, landmark in enumerate(world.landmarks):
            landmark.name = "poi_%d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = ENERGY_RADIUS
            landmark.m_energy = CONFIG["energy"]

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world: CoverageWorld):
        pos_agents= [[[x, x], [x, -x], [-x, -x], [-x, x]] for x in [0.45 * CONFIG["r_comm"]]][0]
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.cover_color = np.array([0.05, 0.25, 0.05])
            agent.comm_color = np.array([0.05, 0.35, 0.05])
            agent.state.p_pos = np.array(pos_agents[i])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.energy = 0.0
            landmark.consume = 0.0
            landmark.done, landmark.just = False, False

        # # set random initial states
        # for agent in world.agents:
        #     agent.state.p_pos = world.np_random.uniform(-1, +1, world.dim_p)
        #     agent.state.p_vel = np.zeros(world.dim_p)
        #     agent.state.c = np.zeros(world.dim_c)
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.state.p_pos = world.np_random.uniform(-1, +1, world.dim_p)
        #     landmark.state.p_vel = np.zeros(world.dim_p)
        
        world.clearStatic()
        
        
        # 随机生成点位，避开障碍区域
        generated = 0
        while generated < len(world.landmarks):
            random_pos = np.random.uniform(-0.8, 0.8, (1, world.dim_p))

            x, y = random_pos[0]
            # 位于障碍区域内
            if world.isInObstacle([x, y]) != -1:
                continue

            world.landmarks[generated].state.p_pos = random_pos[0]
            generated += 1

        world.update_connect()

    def benchmark_data(self, agent: Agent, world: CoverageWorld):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1: Agent, agent2: Agent):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world: CoverageWorld):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        # rew = 0
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)
        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= 1
        # return rew
        rew = 0.0

        # 覆盖奖惩
        for poi in world.landmarks:
            if not poi.done:
                dists = [np.linalg.norm(ag.state.p_pos - poi.state.p_pos) for ag in world.agents]
                rew -= min(dists)
                # 距离poi最近的uav, 二者之间的距离作为负奖励, 该poi的energy_to_cover为乘数
            elif poi.just:
                poi.just = False
                # rew += self.rew_cover
                rew += REWARD["cover"] * poi.consume
                poi.consume = 0.0

        # 全部覆盖完成
        if all([poi.done for poi in world.landmarks]):
            rew += REWARD["done"]

        # 出界惩罚
        for ag in world.agents:
            abs_pos = np.abs(ag.state.p_pos)
            rew -= np.sum(abs_pos[abs_pos > 1] - 1) * 100
            if (abs_pos > 1.2).any():
                rew += REWARD["out_of_bound"]

        # 通信惩罚
        if not world.connect_:
            rew += REWARD["unconnected"]

        # 相互碰撞惩罚
        for i, ag in enumerate(world.agents):
            for j, ag2 in enumerate(world.agents):
                if i < j:
                    dist = np.linalg.norm(ag.state.p_pos - ag2.state.p_pos)
                    if dist < 0.2:
                        rew += REWARD["collision"]

        # 障碍物碰撞惩罚
        for ag in world.agents:
            if world.isInObstacle(ag.state.p_pos) != -1:
                rew += REWARD["collision"]

        return rew

    def observation(self, agent, world: CoverageWorld):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            entity_pos.append([max(entity.m_energy - entity.energy, 0)])

        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + [observation_obstacle])

    def done(self, agent, world):
        for ag in world.agents:
            abs_pos = np.abs(ag.state.p_pos)
            if (abs_pos > 1.5).any():
                return True
        return all([poi.done for poi in world.landmarks])

    def info(self, agent, world):
        pass