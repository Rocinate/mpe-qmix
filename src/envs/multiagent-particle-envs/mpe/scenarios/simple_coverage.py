import numpy as np
from mpe.coverageWord import CoverageWorld
from mpe.core import Landmark, Agent
from mpe.scenario import BaseScenario
import random

# 改为使用左右两点进行表示，这样可以大幅缩小障碍物的表示空间
# 【xmin, ymin, xmax, ymax]，左下角和右上角
obstacle = np.array([
    [0.1, 0.1, 0.3, 0.3],
    [-0.3, -0.3, -0.1, -0.1]
])

obstancle_full = np.array([
    [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]
    for xmin, ymin, xmax, ymax in obstacle
])

# 避免每次observation都需要reshape一次
observation_obstacle = obstacle.reshape(-1)

REWARD = {
    "in_range": 2,
    "cover": 1,
    "done": 5,
    "no_collaps": 2,
    "not_found": 0.1
}

CONFIG = {
    "num_detect": 0,
    "r_comm": 1.0,
    "big_cover": 0.25,
    "small_cover": 0.25,
    "agent_size": 0.02,
    "landmark_size": 0.02,
    "energy": 5.0,
    "fast_speed": 1.0,
    "slow_speed": 1.0
}

class Scenario(BaseScenario):
    def make_world(self):
        # ugly implement, try to check if scared cout can catch this
        print(obstacle)
        print(REWARD, CONFIG)

        num_agents = 4
        num_landmark = 20

        world = CoverageWorld(
            num_agents = num_agents, 
            num_landmark = num_landmark,
            obstacle = obstacle, 
        )

        # set any world properties first
        world.dim_c = 2
        world.collaborative = True

        world.agents = [Agent() for _ in range(num_agents)]
        world.landmarks = [Landmark() for _ in range(num_landmark)]

        # coverage agent    
        for i, agent in enumerate(world.agents):
            agent.name = "agent_%d" % i
            agent.collide = False
            agent.silent = True
            agent.size = CONFIG["agent_size"]
            agent.r_cover = CONFIG["small_cover"]
            agent.r_comm = CONFIG["r_comm"]
            agent.max_speed = CONFIG["slow_speed"]

        # detect agent
        for agent in enumerate(world.agents[-CONFIG["num_detect"]:]):
            agent.r_cover = CONFIG["big_cover"]
            agent.max_speed = CONFIG["fast_speed"]

        for i, landmark in enumerate(world.landmarks):
            landmark.name = "poi_%d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = CONFIG["landmark_size"]
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
        
        # color for detect agents
        for agent in world.agents[-CONFIG["num_detect"]:]:
            agent.color = np.array([0.35, 0.85, 0.35])
            agent.cover_color = np.array([0.05, 0.25, 0.05])
            agent.comm_color = np.array([0.05, 0.35, 0.05])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.energy = 0.0
            landmark.consume = 0.0
            landmark.found = False
            landmark.done, landmark.just = False, False
        
        world.clearStatic()
        
        # 随机生成点位，避开障碍区域
        generated = 0
        while generated < len(world.landmarks):
            random_pos = np.array([random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)])

            if world.isInObstacle(random_pos):
                continue

            world.landmarks[generated].state.p_pos = random_pos
            generated += 1

        world.update_connect()

    def reward(self, agent, world: CoverageWorld):
        rew = 0.0

        # 覆盖奖惩
        for poi in world.landmarks:
            if not poi.found:
                rew -= REWARD["not_found"]
            elif not poi.done:
                dists = [np.linalg.norm(ag.state.p_pos - poi.state.p_pos) for ag in world.agents]
                rew -= min(dists)
                # 距离poi最近的uav, 二者之间的距离作为负奖励, 该poi的energy_to_cover为乘数
            elif poi.just:
                rew += REWARD["cover"] * poi.consume
                poi.consume = 0.0
                poi.just = False

        # 全部覆盖完成
        if all([poi.done for poi in world.landmarks]):
            rew += REWARD["done"]

        # 出界奖励
        if world.outRange == False:
            rew += REWARD["in_range"]

        # 通信惩罚
        # if not world.connect:
        #     rew += REWARD["unconnected"]

        # 无碰撞奖励
        if world.collisionWithOther == False:
            rew += REWARD["no_collaps"]
        
        if world.collisionWithObstacle == False:
            rew += REWARD["no_collaps"]

        # 相互碰撞惩罚
        # for i, ag in enumerate(world.agents):
        #     for j, ag2 in enumerate(world.agents):
        #         if i < j:
        #             dist = np.linalg.norm(ag.state.p_pos - ag2.state.p_pos)
        #             if dist < 0.1:
        #                 rew += REWARD["collision"]

        # 障碍物碰撞惩罚
        # for ag in world.agents:
        #     if world.isInObstacle(ag.state.p_pos):
        #         rew += REWARD["collision"]

        return rew

    def observation(self, agent, world: CoverageWorld):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            if entity.found == False:
                entity_pos.append([0, 0])
                entity_pos.append([entity.m_energy])
            else:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                entity_pos.append([max(entity.m_energy - entity.energy, 0)])

        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + entity_pos + [observation_obstacle])

    def done(self, agent, world):
        for ag in world.agents:
            abs_pos = np.abs(ag.state.p_pos)
            if (abs_pos > 1.2).any():
                return True
        return all([poi.done for poi in world.landmarks])

    # 注入全局信息
    def info(self, agent, world: CoverageWorld):
        overall_info = {}

        overall_info["coverage"] = world.coverage_rate
        overall_info["outRate"] = world.outRange_time / world.episode_length
        overall_info["collisionWithOther"] = world.collisionWithOtherTime / world.episode_length
        overall_info["collisionWithObstacle"] = world.collisionWithObstacleTime / world.episode_length

        return overall_info