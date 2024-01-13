from ast import List
import numpy as np
from mpe.core import Agent, Landmark, World

class CoverageWorld(World):
    def __init__(self, num_agents = 4, num_landmark = 20, comm_r_scale=1.0, comm_force_scale=0.0, obstacle=None):
        super(CoverageWorld, self).__init__()
        self.connect = False
        self.num_agents = num_agents
        self.num_landmark = num_landmark
        self.landmarks: List[Landmark] = []
        self.agents: List[Agent] = []
        self.found = [False for _ in range(num_landmark)]

        # 对连通保持聚合力的修正
        self.contact_force *= comm_force_scale  # 修正拉力倍数，正常情况下都是0
        self.comm_r_scale = comm_r_scale  # 产生拉力的半径 = r_comm * comm_r_scale

        # 距离矩阵，用于连通计算
        self.dist_mat = np.zeros((self.num_agents, self.num_agents))

        # 障碍物信息
        self.obstacle = obstacle

        # 统计信息
        self.episode_length = 0
        self.connect_time = 0
        self.collisionWithObstacleTime = 0
        self.collisionWithOtherTime = 0
        self.outRange_time = 0
        
        # 训练参数
        self.dt = 0.1
        self.delta = 0.01
        self.sensitivity = 2.0

    def clearStatic(self):
        self.connect = False
        self.connect_ = False
        self.episode_length = 0
        self.connect_time = 0
        self.collisionWithObstacleTime = 0
        self.collisionWithOtherTime = 0
        self.outRange_time = 0

    # 假设所有障碍物都是正方形（长方形组成的），异形障碍物可由这两个基本元素组成
    def isInObstacle(self, pos):
        x, y = pos
        result = (self.obstacle[:, 0] < x) & (self.obstacle[:, 2] > x) & (self.obstacle[:, 1] < y) & (self.obstacle[:, 3] > y)
        return np.any(result)

    def step(self):
        # self.update_connect()
        self.update_collision()
        self.episode_length += 1

        p_force = [None for _ in range(len(self.agents))]
        p_force = self.apply_action_force(p_force)

        # tmp_pos = self.check_state(p_force)
        # isconn = self.check_connect(tmp_pos)
        # if not isconn:
        #     if self.contact_force > 0:
        #         p_force = self.apply_connect_force(p_force)

        self.integrate_state(p_force)

        self.update_energy()

    def update_collision(self):
        self.collisionWithOther = False
        self.collisionWithObstacle = False
        self.outRange = False

        # 检查出界
        for ag in self.agents:
            abs_pos = np.abs(ag.state.p_pos)
            if (abs_pos > 1).any():
                self.outRange = True
                self.outRange_time += 1
                break

        # 检查无人机碰撞
        for i, ag in enumerate(self.agents):
            if self.collisionWithOther:
                break
            for j, ag2 in enumerate(self.agents):
                if i < j:
                    dist = np.linalg.norm(ag.state.p_pos - ag2.state.p_pos)
                    if dist < 0.1:
                        self.collisionWithOther = True
                        self.collisionWithOtherTime += 1
                        break

        # 检查障碍物碰撞
        for ag in self.agents:
            if self.isInObstacle(ag.state.p_pos) != -1:
                self.collisionWithObstacle = True
                self.collisionWithObstacleTime += 1
                break

    def update_connect(self):
        self.connect = True

    def apply_action_force(self, p_force):
        for i, agent in enumerate(self.agents):
            p_force[i] = agent.action.u
        return p_force

    def apply_connect_force(self, p_force):
        # 对强连通分支A, 计算其与所有其他强连通分支的距离, 取最短距离, 在此距离的两端产生拉力,
        # 4agent的简化版本:
        # 1) 对没有和其他agent建立连接的孤立agent, 会受到与他最近的agent之间的拉力
        # 2) 若所有agent均有邻居但未达到全连接, 则对当前所有距离里比通信距离大的最小距离添加拉力
        if self.connect_:
            return p_force
        tmp_mat = np.sum(self.adj_mat_, 0)  # 列和为0的agent表示没有连接
        idxs = np.where(tmp_mat == 0)[0]  # idxs 为孤立agent的索引
        # 1) 对孤立agent, 受到与它最近的agent之间的拉力
        if len(idxs) > 0:
            for a in idxs:
                dists = self.dist_mat[a, :]  # 孤立agent与其他agent的距离(与自己的为1e5)
                b = np.argmin(dists)  # 距离孤立agent最近的agent的索引
                [f_a, f_b] = self.get_connect_force(self.agents[a], self.agents[b])
                p_force[a] += f_a
                p_force[b] += f_b
        # 2) 若所有agent均有邻居但未达到全连接, 则对当前所有距离里比通信距离大的最小距离添加拉力
        else:
            idx1 = (self.dist_mat < self.comm_r_scale * 2 * self.agents[0].r_comm)
            self.dist_mat[idx1] = 1e5
            idx2 = np.argmin(self.dist_mat)
            a = idx2 // len(self.agents)
            b = idx2 % len(self.agents)
            [f_a, f_b] = self.get_connect_force(self.agents[a], self.agents[b])
            p_force[a] += f_a
            p_force[b] += f_b
        return p_force

    def get_connect_force(self, agent_a, agent_b):
        if agent_a is agent_b:
            return [0, 0]
        delta_pos = agent_a.state.p_pos - agent_b.state.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_max = (agent_a.r_comm) * self.comm_r_scale
        k = self.contact_margin
        penetration = np.logaddexp(0, (dist - dist_max) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = -force
        force_b = +force
        return [force_a, force_b]

    def integrate_state(self, p_force):
        for i, entity in enumerate(self.agents):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt

            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(
                                                                          entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    # 将p_force输入给系统，输出下一步的位置
    def check_state(self, p_force):
        tmp_vel = [ag.state.p_vel * 1 for ag in self.agents]
        tmp_pos = [ag.state.p_pos * 1 for ag in self.agents]
        for i, agent in enumerate(self.agents):
            tmp_vel[i] = agent.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                tmp_vel += (p_force[i] / agent.mass) * self.dt
            if agent.max_speed is not None:
                speed = np.linalg.norm(tmp_vel[i])
                if speed > agent.max_speed:
                    tmp_vel[i] = tmp_vel[i] / speed * agent.max_speed
            tmp_pos[i] += tmp_vel[i] * self.dt
        return tmp_pos

    def check_connect(self, tmp_pos):
        # 更新邻接矩阵adj_mat和adj_mat_, adj对角线为0, dist对角线为1e5
        self.adj_mat = np.zeros([len(self.agents), len(self.agents)])
        for a, agent_a in enumerate(self.agents):
            for b, agent_b in enumerate(self.agents):
                self.dist_mat[a, b] = np.linalg.norm(tmp_pos[a] - tmp_pos[b])
                if self.dist_mat[a, b] < agent_a.r_comm:
                    self.adj_mat[a, b] = 1
            self.dist_mat[a, a] = 1e5
            self.adj_mat[a, a] = 0
        # 更新connect
        connect_mat = [np.eye(len(self.agents))]
        for _ in range(len(self.agents) - 1):
            connect_mat.append(np.matmul(connect_mat[-1], self.adj_mat))
        return True if (sum(connect_mat) > 0).all() else False

    def update_energy(self):
        num_done = 0

        for poi in self.landmarks:
            if poi.done:
                num_done += 1
            else:
                for agent in self.agents:
                    dist = np.linalg.norm(poi.state.p_pos - agent.state.p_pos)
                    if dist <= agent.r_cover:
                        # poi.energy += (1 - dist / agent.r_cover)  # power随半径线性减少
                        # poi.energy += 1
                        # TODO: 調整覆蓋能量大小，當前最大可以噠到4
                        poi.consume += 2 * np.exp(-dist ** 2 / agent.r_cover ** 2) # 论文公式6, Sc = 2

                if poi.consume > 0.0:
                    # 限制覆蓋能量大小，最大為m_energy
                    poi.energy += poi.consume
                    if poi.energy > poi.m_energy:
                        poi.consume -= poi.energy - poi.m_energy
                        poi.energy = poi.m_energy
                    poi.just = True

                if poi.energy >= poi.m_energy:
                    poi.done = True
                    poi.just = True
                    num_done += 1
                poi.color = np.array([0.25, 0.25 + poi.energy / poi.m_energy * 0.75, 0.25])
        self.coverage_rate = num_done / self.num_landmark
