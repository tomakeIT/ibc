import tensorflow as tf
import pandas as pd
import numpy as np
from data.dataset import load_tfrecord_dataset_sequence, get_shards

# 设置数据路径
dataset_path = "/home/jialeng/ibc/data/particle_gravity/2d_oracle_particle_0.tfrecord"

# 获取数据分片
path_to_shards = get_shards(dataset_path)
print(f"找到 {len(path_to_shards)} 个数据分片: {path_to_shards}")

# 加载序列数据集
sequence_length = 1  # 序列长度
dataset = load_tfrecord_dataset_sequence(
    path_to_shards=path_to_shards,
    seq_len=sequence_length,
    compress_image=True,
    for_rnn=False
)

# 存储所有数据的列表
all_data = []

print("开始读取数据...")
for i, trajectory in enumerate(dataset):
    breakpoint()
    # 提取数据
    step_type = trajectory.step_type.numpy()[0] if hasattr(trajectory.step_type, 'numpy') else trajectory.step_type
    action = trajectory.action.numpy()[0] if hasattr(trajectory.action, 'numpy') else trajectory.action
    reward = trajectory.reward.numpy()[0] if hasattr(trajectory.reward, 'numpy') else trajectory.reward
    discount = trajectory.discount.numpy()[0] if hasattr(trajectory.discount, 'numpy') else trajectory.discount
    
    # 提取观察数据
    observation = trajectory.observation
    pos_agent = observation['pos_agent'].numpy()[0] if hasattr(observation['pos_agent'], 'numpy') else observation['pos_agent']
    pos_first_goal = observation['pos_first_goal'].numpy()[0] if hasattr(observation['pos_first_goal'], 'numpy') else observation['pos_first_goal']
    pos_second_goal = observation['pos_second_goal'].numpy()[0] if hasattr(observation['pos_second_goal'], 'numpy') else observation['pos_second_goal']
    vel_agent = observation['vel_agent'].numpy()[0] if hasattr(observation['vel_agent'], 'numpy') else observation['vel_agent']
    
    # 创建数据行
    data_row = {
        'sample_id': i,
        'step_type': step_type,
        'action_x': action[0],
        'action_y': action[1],
        'reward': reward,
        'discount': discount,
        'pos_agent_x': pos_agent[0],
        'pos_agent_y': pos_agent[1],
        'pos_first_goal_x': pos_first_goal[0],
        'pos_first_goal_y': pos_first_goal[1],
        'pos_second_goal_x': pos_second_goal[0],
        'pos_second_goal_y': pos_second_goal[1],
        'vel_agent_x': vel_agent[0],
        'vel_agent_y': vel_agent[1]
    }
    
    all_data.append(data_row)
    if i >= 1000:
        break

print(f"总共读取了 {len(all_data)} 个样本")

# 转换为DataFrame并保存为CSV
df = pd.DataFrame(all_data)
output_file = "particle_data.csv"
df.to_csv(output_file, index=False)
print(f"数据已保存到 {output_file}")
print(f"CSV文件包含 {len(df)} 行和 {len(df.columns)} 列")
print("列名:", list(df.columns))
