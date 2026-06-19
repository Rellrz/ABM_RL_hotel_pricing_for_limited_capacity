"""
酒店-OTA博弈系统训练脚本

训练酒店和OTA的双层博弈系统
"""

import argparse
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import RL_CONFIG, PATH_CONFIG
import pandas as pd
from src.training.game_trainer import train_game_system, plot_game_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练酒店-OTA博弈系统')
    parser.add_argument('--data', type=str, default='datasets/hotel_bookings.csv',
                       help='酒店预订数据文件路径')
    parser.add_argument('--episodes', type=int, default=50, help='训练轮数')
    parser.add_argument('--mode', type=str, default='simultaneous', 
                       choices=['fixed_ota', 'alternating', 'simultaneous'],
                       help='训练模式')
    parser.add_argument('--commission', type=float, default=0.20, help='OTA佣金率')
    parser.add_argument('--subsidy-ratio-max', type=float, default=0.8, help='最大补贴比例（占佣金的百分比）')
    parser.add_argument('--update-frequency', type=int, default=30, help='CEM参数更新频率（每N天更新一次）')
    parser.add_argument('--booking-window-days', type=int, default=91, help='预订窗口长度（0-90天对应91）')
    parser.add_argument('--decision-buckets', type=str, default='0|1|2-3|4-6|7-13|14-29|30-59|60-90', help='提前期分桶，如: 0|1|2-3|4-6|7-13|14-29|30-59|60-90（为空则自动生成）')
    parser.add_argument('--cem-init-strategy', type=str, default=RL_CONFIG.cem_init_strategy, choices=['midpoint', 'emsrb', 'blended'], help='CEM初始均值策略')
    parser.add_argument('--cem-init-blend-alpha', type=float, default=RL_CONFIG.cem_init_blend_alpha, help='blended初始化时 EMSR-b prior 权重')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("酒店-OTA博弈系统训练")
    print("=" * 60)
    print(f"训练轮数: {args.episodes}")
    print(f"训练模式: {args.mode}")
    print(f"佣金率: {args.commission * 100:.1f}%")
    print(f"补贴比例范围: 0%-{args.subsidy_ratio_max * 100:.0f}%")
    print(f"CEM更新频率: 每{args.update_frequency}天更新一次")
    print(f"预订窗口: 0-{args.booking_window_days - 1}天")
    print(f"CEM初始化策略: {args.cem_init_strategy}")
    if args.cem_init_strategy == 'blended':
        print(f"CEM初始化混合系数: {args.cem_init_blend_alpha:.2f}")
    if str(args.decision_buckets).strip():
        print(f"提前期分桶: {args.decision_buckets}")
    
    # 更新配置
    RL_CONFIG.commission_rate = args.commission
    RL_CONFIG.subsidy_ratio_max = args.subsidy_ratio_max
    RL_CONFIG.cem_init_strategy = args.cem_init_strategy
    RL_CONFIG.cem_init_blend_alpha = args.cem_init_blend_alpha
    
    # 加载数据
    print("\n=== 加载历史数据 ===")
    historical_data = pd.read_csv(args.data)
    historical_data = historical_data[historical_data['hotel'] == 'City Hotel'].copy()
    print(f"数据加载完成，共 {len(historical_data)} 条记录") 
    
    # 训练博弈系统
    hotel_agent, ota_agent, rewards_hotel, rewards_ota, episode_info = train_game_system(
        historical_data=historical_data,
        episodes=args.episodes,
        training_mode=args.mode,
        update_frequency=args.update_frequency,
        booking_window_days=args.booking_window_days,
        decision_buckets=args.decision_buckets,
    )
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存模型参数
    hotel_agent.save()
    #ota_agent.save()

    # 绘制结果图表
    figure_dir = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    figure_path = os.path.join(figure_dir, f'game_results_{timestamp}.png')
    
    plot_game_results(rewards_hotel, rewards_ota, episode_info, figure_path)
    
    # 保存训练数据
    df = pd.DataFrame(episode_info)
    data_path = os.path.join(PATH_CONFIG.models_dir, f'training_data_{timestamp}.csv')
    df.to_csv(data_path, index=False)
    print(f"\n训练数据已保存到: {data_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
