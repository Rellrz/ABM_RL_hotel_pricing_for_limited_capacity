"""
酒店-OTA博弈系统训练脚本

训练酒店和OTA的双层博弈系统
"""

import argparse
import os
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import glob

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import RL_CONFIG, PATH_CONFIG, ENV_CONFIG
from src.training.game_trainer import train_game_system, plot_game_results


def _run_one_capacity(capacity: int, args_dict: dict) -> dict:
    try:
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
        os.environ.setdefault('MPLBACKEND', 'Agg')

        import pandas as pd

        historical_data = pd.read_csv(args_dict['data'])
        historical_data = historical_data[historical_data['hotel'] == 'City Hotel'].copy()

        RL_CONFIG.commission_rate = args_dict['commission']
        RL_CONFIG.subsidy_ratio_max = args_dict['subsidy_ratio_max']

        ENV_CONFIG.initial_inventory = capacity

        hotel_agent, ota_agent, rewards_hotel, rewards_ota, episode_info = train_game_system(
            historical_data=historical_data,
            episodes=args_dict['episodes'],
            training_mode=args_dict['mode'],
            update_frequency=args_dict['update_frequency'],
            booking_window_days=args_dict['booking_window_days'],
            decision_buckets=args_dict['decision_buckets'],
        )

        run_id = f"cap{capacity}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"

        model_paths = {}
        if hasattr(hotel_agent, 'cem_joint') and hotel_agent.cem_joint is not None and hasattr(hotel_agent.cem_joint, 'save_model'):
            prefix = f'hotel_joint_{run_id}'
            hotel_agent.cem_joint.save_model(prefix)
            candidates = glob.glob(os.path.join(PATH_CONFIG.models_dir, f'{prefix}_agent_*.json'))
            if candidates:
                model_paths['hotel_joint'] = max(candidates, key=os.path.getmtime)
        if hasattr(hotel_agent, 'cem_online') and hasattr(hotel_agent.cem_online, 'save_model'):
            prefix = f'hotel_online_{run_id}'
            hotel_agent.cem_online.save_model(prefix)
            candidates = glob.glob(os.path.join(PATH_CONFIG.models_dir, f'{prefix}_agent_*.json'))
            if candidates:
                model_paths['hotel_online'] = max(candidates, key=os.path.getmtime)
        if hasattr(hotel_agent, 'cem_offline') and hasattr(hotel_agent.cem_offline, 'save_model'):
            prefix = f'hotel_offline_{run_id}'
            hotel_agent.cem_offline.save_model(prefix)
            candidates = glob.glob(os.path.join(PATH_CONFIG.models_dir, f'{prefix}_agent_*.json'))
            if candidates:
                model_paths['hotel_offline'] = max(candidates, key=os.path.getmtime)
        if hasattr(ota_agent, 'save_model'):
            prefix = f'ota_{run_id}'
            ota_agent.save_model(prefix)
            candidates = glob.glob(os.path.join(PATH_CONFIG.models_dir, f'{prefix}_agent_*.json'))
            if candidates:
                model_paths['ota'] = max(candidates, key=os.path.getmtime)

        figure_dir = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
        os.makedirs(figure_dir, exist_ok=True)
        figure_path = os.path.join(figure_dir, f'game_results_capacity_{run_id}.png')
        plot_game_results(rewards_hotel, rewards_ota, episode_info, figure_path)

        df = pd.DataFrame(episode_info)
        data_path = os.path.join(PATH_CONFIG.models_dir, f'training_data_capacity_{run_id}.csv')
        df.to_csv(data_path, index=False)

        return {
            'capacity': capacity,
            'data_path': data_path,
            'figure_path': figure_path,
            'model_paths': model_paths,
            'status': 'ok',
        }
    except Exception as e:
        return {
            'capacity': capacity,
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练酒店-OTA博弈系统')
    parser.add_argument('--data', type=str, default='datasets/hotel_bookings.csv',
                       help='酒店预订数据文件路径')
    parser.add_argument('--episodes', type=int, default=250, help='训练轮数')
    parser.add_argument('--mode', type=str, default='simultaneous', 
                       choices=['fixed_ota', 'alternating', 'simultaneous'],
                       help='训练模式')
    parser.add_argument('--commission', type=float, default=0.20, help='OTA佣金率')
    parser.add_argument('--subsidy-ratio-max', type=float, default=0.8, help='最大补贴比例（占佣金的百分比）')
    parser.add_argument('--update-frequency', type=int, default=30, help='CEM参数更新频率（每N天更新一次）')
    parser.add_argument('--booking-window-days', type=int, default=91, help='预订窗口长度（0-90天对应91）')
    parser.add_argument('--decision-buckets', type=str, default='0|1|2-3|4-6|7-13|14-29|30-59|60-90', help='提前期分桶，如: 0|1|2-3|4-6|7-13|14-29|30-59|60-90（为空则自动生成）')
    parser.add_argument('--workers', type=int, default=4, help='并行进程数（建议CPU: 2-4，GPU: 1）')
    
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
    if str(args.decision_buckets).strip():
        print(f"提前期分桶: {args.decision_buckets}")
    
    hotel_capacity = [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
    results = {}
    results_fig = {}
    results_models = {}

    args_dict = {
        'data': args.data,
        'episodes': args.episodes,
        'mode': args.mode,
        'commission': args.commission,
        'subsidy_ratio_max': args.subsidy_ratio_max,
        'update_frequency': args.update_frequency,
        'booking_window_days': args.booking_window_days,
        'decision_buckets': args.decision_buckets,
    }

    if int(args.workers) <= 1:
        for capacity in hotel_capacity:
            result = _run_one_capacity(capacity, args_dict)
            if result['status'] == 'ok':
                print(f"[OK] capacity={capacity} csv={result['data_path']} fig={result['figure_path']}")
                results[capacity] = result['data_path']
                results_fig[capacity] = result['figure_path']
                results_models[capacity] = result.get('model_paths', {})
            else:
                print(f"[ERROR] capacity={capacity} error={result['error']}")
                print(result['traceback'])
    else:
        max_workers = min(int(args.workers), len(hotel_capacity))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one_capacity, c, args_dict): c for c in hotel_capacity}
            for fut in as_completed(futures):
                result = fut.result()
                capacity = result.get('capacity')
                if result.get('status') == 'ok':
                    print(f"[OK] capacity={capacity} csv={result['data_path']} fig={result['figure_path']}")
                    results[capacity] = result['data_path']
                    results_fig[capacity] = result['figure_path']
                    results_models[capacity] = result.get('model_paths', {})
                else:
                    print(f"[ERROR] capacity={capacity} error={result.get('error')}")
                    print(result.get('traceback', ''))
    
    results_dir = os.path.join(PROJECT_ROOT, 'outputs', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    results_path = os.path.join(results_dir, f'capacity_to_csv_{results_timestamp}.json')
    figures_path = os.path.join(results_dir, f'capacity_to_figure_{results_timestamp}.json')
    models_path = os.path.join(results_dir, f'capacity_to_model_{results_timestamp}.json')
    import json
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in results.items()}, f, ensure_ascii=False, indent=2)
    with open(figures_path, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in results_fig.items()}, f, ensure_ascii=False, indent=2)
    with open(models_path, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in results_models.items()}, f, ensure_ascii=False, indent=2)
    print(f"\n[Summary] capacity→csv 已保存: {results_path}")
    print(f"[Summary] capacity→figure 已保存: {figures_path}")
    print(f"[Summary] capacity→model 已保存: {models_path}")
    print(f"[Summary] capacity→csv: {results}")

    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
