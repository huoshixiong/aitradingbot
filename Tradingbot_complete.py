#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ccxt
import pandas as pd
import numpy as np
import time
import json
import logging
import logging.handlers
import os
import signal
import traceback
import hashlib
import fcntl
import re
import threading
import gc
import sys
import math
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv
from collections import deque
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import requests
from typing import Dict, List, Optional, Tuple, Any

# ========= 环境变量热重载 =========
load_dotenv(override=True)

# ========= 安全配置验证 =========
def validate_config():
    """严格的配置验证"""
    api_key = os.getenv('GATEIO_KEY')
    secret = os.getenv('GATEIO_SECRET')
    
    # 正确检查空值
    if not api_key or not secret:
        logging.critical('🚨 API密钥为空，程序退出')
        raise SystemExit('API密钥未配置')
    
    # 检查是否为默认占位符
    if 'YOUR_' in api_key or 'YOUR_' in secret:
        logging.critical('🚨 检测到默认API密钥，请配置真实密钥')
        raise SystemExit('使用默认API密钥')
    
    # 验证密钥格式
    if len(api_key) < 10 or len(secret) < 10:
        logging.critical('🚨 API密钥格式异常')
        raise SystemExit('API密钥格式错误')
    
    return api_key, secret

# 执行配置验证
API_KEY, SECRET = validate_config()

# ========= 基础配置 =========
CFG = {
    'apiKey': API_KEY,
    'secret': SECRET,
    'sandbox': False,  # Gate.io 测试网支持有限，默认关闭
    'symbols': ['BTC/USDT:USDT', 'ETH/USDT:USDT'],  # Gate.io 永续合约格式
    'timeframes': ['15m', '1h'],
    'main_timeframe': '15m',
    'lev': 15,
    'max_equity_risk': float(os.getenv('MAX_RISK', 0.25)),
    'base_trend_unit': 100,
    'base_grid_unit': 50,
    'position_file': 'ultimate_pos_ai_gate.json',
    'log_file': 'ultimate_pro_ai_gate.log',
    'performance_file': 'performance_metrics_gate.json',
    'loop_sec': 10,
    'vol_filter': 0.70,
    'funding_limit': 0.001,  # 从 0.0003 提高到 0.001 (0.1%)
    'funding_time_limit_h': 1,  # 从 4小时减少到 1小时
    
    # AI增强配置
    'ai_optimization': {
        'enabled': True,
        'min_trades_for_optimization': 8,
        'max_parameter_change_ratio': 0.3,
        'backtest_lookback_days': 30,
        'validation_threshold': 0.8,
        'base_interval_hours': 12,
        'high_frequency_interval': 6,
        'low_frequency_interval': 24,
        'position_optimization_enabled': True,
        'weight_optimization_interval_hours': 6,
    },
    
    'circuit_breaker': {'max_drawdown': 0.15, 'daily_loss_limit': 0.10},
    'min_notional': 5,  # 降低最小交易额
    'min_position_ratio': 0.3,
    'max_position_ratio': 2.5,
    
    # AI仓位管理配置
    'ai_position_management': {
        'enabled': True,
        'max_single_risk': 0.12,  # 从 0.08 提高到 0.12
        'max_symbol_risk': 0.6,   # 从 0.4 提高到 0.6
        'max_total_risk': 1.2,    # 从 0.8 提高到 1.2
        'volatility_adjustment': True,
        'performance_feedback': True,
    },
    
    # 资金量分级配置
    'capital_tiers': {
        'micro': {'min': 50, 'max': 1000, 'base_risk': 0.04},
        'small': {'min': 1000, 'max': 10000, 'base_risk': 0.03},
        'medium': {'min': 10000, 'max': 100000, 'base_risk': 0.025},
        'large': {'min': 100000, 'max': 1000000, 'base_risk': 0.02},
        'institutional': {'min': 1000000, 'max': float('inf'), 'base_risk': 0.015}
    },
    
    # 真实交易环境配置
    'realistic_trading': {
        'enable_dynamic_slippage': True,
        'max_slippage': 0.05,
        'funding_avoid_hours': 4,
        'min_position_value': 5,  # 降低最小仓位价值
        'volatility_adjusted_risk': True,
    },
    
    # 高级风险管理
    'advanced_risk_management': {
        'max_daily_trades': 50,  # 从 20 提高到 50
        'correlation_threshold': 0.8,  # 从 0.7 提高到 0.8
        'liquidity_threshold': 0.05,   # 从 0.1 降低到 0.05
    },
    
    # 交易成本配置
    'trading_costs': {
        'taker_fee': 0.0005,  # Gate.io taker费率 0.05%
        'maker_fee': 0.0002,  # Gate.io maker费率 0.02%
        'base_slippage': 0.0005,
        'max_slippage': 0.01,
    },
    
    # 风险预算配置
    'risk_management': {
        'base_risk_per_trade': 0.01,
        'max_position_risk': 0.4,
        'volatility_adjustment': True,
        'max_daily_loss': 0.05,
    },
    
    # 权重系统配置
    'weight_system': {
        'weights': {
            'trend_strength': 0.25,
            'momentum': 0.20,
            'volatility': 0.15,
            'rsi_signal': 0.15,
            'bb_signal': 0.10,
            'volume_confirmation': 0.08,
            'market_regime': 0.07
        },
        'thresholds': {
            'open_long': 0.65,
            'open_short': 0.65,
            'close_position': 0.35,
            'strong_signal': 0.80
        }
    }
}

# ========= 安全日志配置 =========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.handlers.TimedRotatingFileHandler(
            CFG['log_file'], when='midnight', backupCount=15, encoding='utf-8')
    ]
)
LOG = logging.getLogger('UltimateProAI_Gate')

# ========= 权重交易系统 =========
class WeightedTradingSystem:
    """权重交易系统 - 集成配置管理"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.condition_weights = self._initialize_weights()
        self.thresholds = self._initialize_thresholds()
        self.performance_history = []
        
    def _initialize_weights(self):
        """从配置管理器初始化权重"""
        default_weights = {
            'trend_strength': 0.25,
            'momentum': 0.20,
            'volatility': 0.15,
            'rsi_signal': 0.15,
            'bb_signal': 0.10,
            'volume_confirmation': 0.08,
            'market_regime': 0.07
        }
        
        # 尝试从配置获取权重，否则使用默认值
        weights = {}
        for condition in default_weights.keys():
            config_key = f'weight_system.weights.{condition}'
            weights[condition] = self.config_manager.get_config_value(config_key, default_weights[condition])
        
        # 归一化权重
        total = sum(weights.values())
        if total > 0:
            for condition in weights:
                weights[condition] /= total
        
        LOG.info(f"权重系统初始化完成: {weights}")
        return weights
    
    def _initialize_thresholds(self):
        """从配置管理器初始化阈值"""
        return {
            'open_long': self.config_manager.get_config_value('weight_system.thresholds.open_long', 0.65),
            'open_short': self.config_manager.get_config_value('weight_system.thresholds.open_short', 0.65),
            'close_position': self.config_manager.get_config_value('weight_system.thresholds.close_position', 0.35),
            'strong_signal': self.config_manager.get_config_value('weight_system.thresholds.strong_signal', 0.80)
        }

# ========= AI权重优化器 =========
class AIWeightOptimizer:
    """AI权重优化器 - 修复逻辑问题"""
    
    def __init__(self, weight_system, ai_optimizer):
        self.weight_system = weight_system
        self.ai_optimizer = ai_optimizer
        self.optimization_history = []
        self.last_optimization_time = 0
    
    def optimize_weights(self, performance_data):
        """优化权重配置 - 返回是否成功"""
        if len(performance_data) < self.weight_system.config_manager.get_config_value(
            'ai_optimization.min_trades_for_optimization', 20):
            LOG.debug("交易数据不足，跳过权重优化")
            return False
        
        # 检查优化间隔
        current_time = time.time()
        optimization_interval = self.weight_system.config_manager.get_config_value(
            'ai_optimization.weight_optimization_interval_hours', 6) * 3600
        
        if current_time - self.last_optimization_time < optimization_interval:
            return False
        
        # 分析各条件的表现
        condition_performance = self._analyze_condition_performance(performance_data)
        
        # 使用AI调整权重
        new_weights = self._ai_adjust_weights(condition_performance)
        
        if new_weights:
            old_weights = self.weight_system.condition_weights.copy()
            self.weight_system.condition_weights = new_weights
            self.last_optimization_time = current_time
            
            self.optimization_history.append({
                'timestamp': datetime.now().isoformat(),
                'old_weights': old_weights,
                'new_weights': new_weights,
                'performance_data': performance_data[-20:],
                'condition_performance': condition_performance
            })
            
            LOG.info("✅ AI权重优化完成")
            self._log_weight_changes(old_weights, new_weights)
            return True
        
        return False
    
    def _log_weight_changes(self, old_weights, new_weights):
        """记录权重变化"""
        LOG.info("📊 权重变化详情:")
        for condition in old_weights:
            old_val = old_weights[condition]
            new_val = new_weights[condition]
            change = (new_val - old_val) / old_val * 100 if old_val > 0 else 0
            LOG.info(f"  {condition}: {old_val:.3f} → {new_val:.3f} ({change:+.1f}%)")
    
    def _analyze_condition_performance(self, performance_data):
        """分析各条件的历史表现"""
        condition_stats = {}
        
        for condition in self.weight_system.condition_weights.keys():
            # 分析该条件在成功交易中的出现频率
            successful_trades_with_condition = [
                trade for trade in performance_data 
                if trade.get('pnl', 0) > 0 and trade.get('conditions', {}).get(condition, 0) > 0.5
            ]
            
            total_trades_with_condition = [
                trade for trade in performance_data 
                if trade.get('conditions', {}).get(condition, 0) > 0.5
            ]
            
            win_rate = len(successful_trades_with_condition) / len(total_trades_with_condition) if total_trades_with_condition else 0
            
            condition_stats[condition] = {
                'win_rate': win_rate,
                'frequency': len(total_trades_with_condition) / len(performance_data) if performance_data else 0,
                'avg_score': np.mean([trade.get('conditions', {}).get(condition, 0) for trade in performance_data])
            }
        
        return condition_stats
    
    def _ai_adjust_weights(self, condition_stats):
        """修复权重归一化"""
        new_weights = self.weight_system.condition_weights.copy()
        
        total_win_rate = sum(stats['win_rate'] for stats in condition_stats.values())
        if total_win_rate == 0:
            return None  # 无法优化
        
        for condition, stats in condition_stats.items():
            # 基于相对胜率调整权重
            relative_performance = stats['win_rate'] / total_win_rate if total_win_rate > 0 else 1.0
            
            # 胜率高且频率适中的条件增加权重
            if stats['win_rate'] > 0.6 and 0.2 < stats['frequency'] < 0.8:
                adjustment = 1.2
            # 胜率低的条件降低权重
            elif stats['win_rate'] < 0.4:
                adjustment = 0.8
            else:
                adjustment = 1.0
            
            # 应用调整
            new_weights[condition] *= adjustment
        
        # 确保权重归一化
        total = sum(new_weights.values())
        if total > 0:
            for condition in new_weights:
                new_weights[condition] /= total
        
        # 验证权重变化是否显著
        max_change = max(abs(new_weights[c] - self.weight_system.condition_weights[c]) 
                        for c in new_weights)
        
        if max_change < 0.05:  # 变化小于5%，认为优化无效
            return None
        
        return new_weights

# ========= AI激进度调节系统 =========
class AggressionController:
    """AI激进度控制器"""
    
    def __init__(self):
        self.current_level = 5
        self.level_file = 'aggression_level_gate.txt'
        self.last_level = 5
        self.last_check_time = 0
        self._lock = threading.RLock()
        self.safety_limits = {
            'max_equity_risk': 0.4,
            'max_single_risk': 0.15,
            'max_total_risk': 1.0,
            'max_position_ratio': 3.5,
        }
        
        # 初始化激进度乘数 - 更大的差异
        self.aggression_multipliers = {
            1: {'risk_multiplier': 0.3, 'weight_multiplier': 0.5, 'threshold_adjust': 0.2},
            2: {'risk_multiplier': 0.5, 'weight_multiplier': 0.7, 'threshold_adjust': 0.15},
            3: {'risk_multiplier': 0.7, 'weight_multiplier': 0.85, 'threshold_adjust': 0.1},
            4: {'risk_multiplier': 0.85, 'weight_multiplier': 0.95, 'threshold_adjust': 0.05},
            5: {'risk_multiplier': 1.0, 'weight_multiplier': 1.0, 'threshold_adjust': 0.0},
            6: {'risk_multiplier': 1.3, 'weight_multiplier': 1.15, 'threshold_adjust': -0.05},
            7: {'risk_multiplier': 1.7, 'weight_multiplier': 1.35, 'threshold_adjust': -0.1},
            8: {'risk_multiplier': 2.2, 'weight_multiplier': 1.6, 'threshold_adjust': -0.15},
            9: {'risk_multiplier': 3.0, 'weight_multiplier': 2.0, 'threshold_adjust': -0.2},
            10: {'risk_multiplier': 4.0, 'weight_multiplier': 2.5, 'threshold_adjust': -0.25}
        }

    def set_aggression_level(self, level):
        if 1 <= level <= 10:
            with self._lock:
                self.last_level = self.current_level
                self.current_level = level
                self._save_level_to_file()
                LOG.info(f"🎛️ AI激进度已切换到级别 {level}")
            return True
        return False

    def has_level_changed(self):
        current_time = time.time()
        if current_time - self.last_check_time < 30:
            return False
        
        self.last_check_time = current_time
        try:
            if os.path.exists(self.level_file):
                with open(self.level_file, 'r') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    content = f.read().strip()
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    if content.isdigit():
                        new_level = int(content)
                        if new_level != self.current_level and 1 <= new_level <= 10:
                            self.last_level = self.current_level
                            self.current_level = new_level
                            LOG.info(f"🎛️ 检测到激进度变化: {self.last_level} -> {self.current_level}")
                            return True
        except Exception as e:
            LOG.error(f"检查激进度变化失败: {e}")
        return False

    def _save_level_to_file(self):
        try:
            with open(self.level_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(str(self.current_level))
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            LOG.error(f"保存激进度级别失败: {e}")

    def get_current_config(self):
        return self._get_config_for_level(self.current_level)

    def _get_config_for_level(self, level):
        aggression_config = self.aggression_multipliers[level]
        
        descriptions = {
            1: "🏛️ 极度保守 - 最大安全优先",
            2: "🛡️ 保守 - 安全优先", 
            3: "🎯 稳健 - 平衡偏安全",
            4: "⚖️ 适中 - 风险收益平衡",
            5: "🔍 平衡 - 原系统设置",
            6: "💹 积极 - 适度激进",
            7: "🚀 激进 - 机会优先",
            8: "🔥 高度激进 - 最大化收益", 
            9: "⚡ 极度激进 - 高风险高回报",
            10: "🎲 赌博模式 - 最大风险"
        }
        
        return {
            'risk_multiplier': aggression_config['risk_multiplier'],
            'position_multiplier': aggression_config['weight_multiplier'],  # 使用weight_multiplier作为position_multiplier
            'optimization_aggressiveness': min(2.0, 0.5 + aggression_config['risk_multiplier'] * 0.5),
            'filters_strictness': max(0.3, 2.0 - aggression_config['risk_multiplier'] * 0.4),
            'description': descriptions.get(level, f"级别 {level}"),
            'threshold_adjust': aggression_config['threshold_adjust'],
            'weight_multiplier': aggression_config['weight_multiplier']
        }

    def validate_aggression_level(self, level, current_equity):
        if not 1 <= level <= 10:
            return False
            
        config = self._get_config_for_level(level)
        proposed_risk = 0.25 * config['risk_multiplier']
        
        if proposed_risk > self.safety_limits['max_equity_risk']:
            LOG.warning(f"风险超出安全边界: {proposed_risk:.1%} > {self.safety_limits['max_equity_risk']:.1%}")
            return False
            
        if current_equity < 1000 and level > 7:
            LOG.warning("资金量过小时不建议使用高激进度")
            return False
            
        return True

    def get_detailed_config(self, level):
        """获取详细的配置报告"""
        config = self._get_config_for_level(level)
        aggression_config = self.aggression_multipliers[level]
        
        report = {
            'level': level,
            'description': config['description'],
            'risk_parameters': {
                'max_equity_risk': f"{0.25 * config['risk_multiplier']:.1%}",
                'max_single_risk': f"{0.08 * config['risk_multiplier']:.1%}",
                'max_symbol_risk': f"{0.4 * config['risk_multiplier']:.1%}",
                'max_total_risk': f"{0.8 * config['risk_multiplier']:.1%}"
            },
            'position_parameters': {
                'base_trend_unit': int(100 * config['position_multiplier']),
                'base_grid_unit': int(50 * config['position_multiplier']),
                'max_position_ratio': f"{2.5 * config['position_multiplier']:.1f}x"
            },
            'weight_parameters': {
                'weight_multiplier': f"{aggression_config['weight_multiplier']:.1f}x",
                'threshold_adjustment': f"{aggression_config['threshold_adjust']:+.2f}",
                'open_threshold': f"{0.65 + aggression_config['threshold_adjust']:.2f}"
            },
            'optimization_parameters': {
                'aggressiveness': f"{config['optimization_aggressiveness']:.1f}x",
                'max_parameter_change': f"{min(0.6, 0.3 * config['optimization_aggressiveness']):.1%}",
                'validation_threshold': f"{max(0.5, 0.8 / config['optimization_aggressiveness']):.2f}"
            },
            'filter_parameters': {
                'strictness': f"{config['filters_strictness']:.1f}x",
                'funding_avoid_hours': f"{min(8, 4 * config['filters_strictness']):.0f}",
                'vol_filter': f"{min(0.9, 0.7 * config['filters_strictness']):.2f}"
            }
        }
        
        return report

class ThreadSafeConfigManager:
    """线程安全的配置管理器"""
    
    def __init__(self, aggression_controller):
        self.aggression_controller = aggression_controller
        self._lock = threading.RLock()
        self._config_overrides = {}
    
    def update_config_overrides(self):
        with self._lock:
            aggression_config = self.aggression_controller.get_current_config()
            self._config_overrides = self._create_safe_overrides(aggression_config)
    
    def _create_safe_overrides(self, aggression_config):
        safe_overrides = {}
        risk_multiplier = min(aggression_config['risk_multiplier'], 4.0)  # 提高上限
        position_multiplier = min(aggression_config['position_multiplier'], 2.5)
        
        safe_overrides['max_equity_risk'] = min(0.4, 0.25 * risk_multiplier)
        safe_overrides['max_single_risk'] = min(0.15, 0.08 * risk_multiplier)
        safe_overrides['max_symbol_risk'] = min(0.6, 0.4 * risk_multiplier)
        safe_overrides['max_total_risk'] = min(1.2, 0.8 * risk_multiplier)
        safe_overrides['base_trend_unit'] = 100 * position_multiplier
        safe_overrides['base_grid_unit'] = 50 * position_multiplier
        safe_overrides['max_position_ratio'] = min(3.5, 2.5 * position_multiplier)
        
        aggression = min(aggression_config['optimization_aggressiveness'], 2.0)
        safe_overrides['ai_optimization'] = {
            'max_parameter_change_ratio': min(0.6, 0.3 * aggression),
            'base_interval_hours': max(2, 12 / aggression),
            'validation_threshold': max(0.5, 0.8 / aggression)
        }
        
        strictness = aggression_config['filters_strictness']
        safe_overrides['realistic_trading'] = {
            'funding_avoid_hours': min(8, 4 * strictness),
            'min_position_value': max(2, 10 / strictness)  # 进一步降低最小仓位
        }
        
        safe_overrides['vol_filter'] = min(0.9, 0.7 * strictness)
        safe_overrides['circuit_breaker'] = {
            'max_drawdown': min(0.3, 0.15 * (2 - strictness)),
            'daily_loss_limit': min(0.2, 0.10 * (2 - strictness))
        }
        
        return safe_overrides
    
    def get_config_value(self, key_path, default=None):
        """获取配置值，优先返回激进度调整后的值"""
        with self._lock:
            try:
                keys = key_path.split('.')
                current_value = self._config_overrides
                
                # 首先在覆盖配置中查找
                for key in keys:
                    if isinstance(current_value, dict) and key in current_value:
                        current_value = current_value[key]
                    else:
                        # 如果在覆盖配置中找不到，回退到原始CFG
                        current_value = CFG
                        for k in keys:
                            if isinstance(current_value, dict) and k in current_value:
                                current_value = current_value[k]
                            else:
                                return default
                        break
                
                if current_value is None:
                    return default
                return current_value
            except Exception as e:
                LOG.debug(f"获取配置{key_path}失败: {e}")
                
                # 回退到直接从CFG获取
                try:
                    keys = key_path.split('.')
                    current_value = CFG
                    for key in keys:
                        current_value = current_value[key]
                    return current_value
                except:
                    return default

class CommandReceiver:
    """命令接收器"""
    
    def __init__(self, aggression_controller, config_manager):
        self.aggression_controller = aggression_controller
        self.config_manager = config_manager
        self.command_file = 'ai_command_gate.txt'
        self.running = True
    
    def start_listening(self):
        def listen_loop():
            while self.running:
                self._check_commands()
                time.sleep(10)
        
        thread = threading.Thread(target=listen_loop, daemon=True)
        thread.start()
        LOG.info("🎛️ AI指令接收器已启动")
    
    def _check_commands(self):
        try:
            if not os.path.exists(self.command_file):
                return
            
            with open(self.command_file, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                content = f.read().strip()
                if content:
                    self._process_command(content)
                    f.seek(0)
                    f.truncate()
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            LOG.error(f"命令处理错误: {e}")
    
    def _process_command(self, command):
        command = command.upper().strip()
        if command.isdigit():
            level = int(command)
            current_equity = self._get_current_equity()
            if self.aggression_controller.validate_aggression_level(level, current_equity):
                success = self.aggression_controller.set_aggression_level(level)
                if success:
                    self.config_manager.update_config_overrides()
                    LOG.info(f"✅ 已应用激进度级别 {level}")
                    self._show_detailed_status()
            else:
                LOG.error(f"❌ 无效指令或安全验证失败: {command}")
        elif command == 'STATUS':
            self._show_detailed_status()
        elif command == 'HELP':
            self._show_help()
        else:
            LOG.error(f"❌ 未知指令: {command}")
    
    def _get_current_equity(self):
        try:
            # 这里应该从交易所获取真实权益，简化实现
            return 1000
        except:
            return 1000

    def _show_detailed_status(self):
        level = self.aggression_controller.current_level
        detailed_config = self.aggression_controller.get_detailed_config(level)
        
        LOG.info(f"\n=== 🎛️ AI激进度状态详情 ===")
        LOG.info(f"当前级别: {level}/10 - {detailed_config['description']}")
        
        LOG.info(f"\n📊 风险参数:")
        for param, value in detailed_config['risk_parameters'].items():
            LOG.info(f"  {param}: {value}")
        
        LOG.info(f"\n💰 仓位参数:")
        for param, value in detailed_config['position_parameters'].items():
            LOG.info(f"  {param}: {value}")
        
        LOG.info(f"\n⚖️ 权重参数:")
        for param, value in detailed_config['weight_parameters'].items():
            LOG.info(f"  {param}: {value}")
        
        LOG.info(f"\n🔧 优化参数:")
        for param, value in detailed_config['optimization_parameters'].items():
            LOG.info(f"  {param}: {value}")
        
        LOG.info(f"\n🛡️ 过滤参数:")
        for param, value in detailed_config['filter_parameters'].items():
            LOG.info(f"  {param}: {value}")
        
        LOG.info("============================\n")
    
    def _show_help(self):
        help_text = """
🤖 AI激进度调节系统 - 命令帮助:

数字命令:
1 - 🏛️ 极度保守 (最大安全)
2 - 🛡️ 保守 (安全优先) 
3 - 🎯 稳健 (平衡偏安全)
4 - ⚖️ 适中 (风险收益平衡)
5 - 🔍 平衡 (原系统设置)
6 - 💹 积极 (适度激进)
7 - 🚀 激进 (机会优先)
8 - 🔥 高度激进 (最大化收益)
9 - ⚡ 极度激进 (高风险高回报)
10 - 🎲 赌博模式 (最大风险)

状态命令:
STATUS - 显示当前状态
HELP - 显示此帮助

使用方法:
1. 编辑 ai_command_gate.txt 文件
2. 输入数字 1-10 或命令
3. 系统将在10秒内响应
"""
        LOG.info(help_text)
    
    def stop_listening(self):
        self.running = False

# ========= 精度计算辅助类 =========
class PrecisionHelper:
    """安全的精度计算工具"""
    
    @staticmethod
    def safe_float_operation(amount_usdt, price, symbol, exchange):
        try:
            amount_decimal = Decimal(str(amount_usdt)) / Decimal(str(price))
            market = exchange.market(symbol)
            amount_precision = market['precision']['amount']
            quantized = amount_decimal.quantize(
                Decimal(f"1e-{amount_precision}"),
                rounding=ROUND_DOWN
            )
            result = float(quantized)
            if result <= 0 or result > market['limits']['amount']['max']:
                LOG.error(f'计算数量异常: {result}')
                return None
            return result
        except Exception as e:
            LOG.error(f'精度计算失败: {e}')
            return None

# ========= 真实滑点模型 =========
class RealisticSlippageModel:
    """真实滑点模型"""
    
    def __init__(self, exchange, config_manager):
        self.ex = exchange
        self.config_manager = config_manager
        self.symbol_liquidity = {}
    
    def calculate_dynamic_slippage(self, symbol, order_size, side):
        try:
            orderbook = retry(self.ex.fetch_order_book, symbol, limit=20)
            if side == 'buy':
                return self._calculate_buy_slippage(orderbook['asks'], order_size)
            else:
                return self._calculate_sell_slippage(orderbook['bids'], order_size)
        except Exception as e:
            LOG.warning(f'滑点计算失败，使用保守估计: {e}')
            return self.config_manager.get_config_value('trading_costs.base_slippage', 0.0005)
    
    def _calculate_buy_slippage(self, asks, order_size):
        total_cost = 0
        filled_size = 0
        base_slippage = 0
        base_slippage_config = self.config_manager.get_config_value('trading_costs.base_slippage', 0.0005)
        max_slippage_config = self.config_manager.get_config_value('trading_costs.max_slippage', 0.01)
        
        for i, (price, volume) in enumerate(asks):
            price = float(price)
            volume = float(volume)
            if filled_size >= order_size:
                break
            available = min(volume, order_size - filled_size)
            total_cost += price * available
            filled_size += available
            if i >= 3:
                base_slippage += 0.0005
        
        if filled_size == 0:
            return max_slippage_config
        
        avg_price = total_cost / filled_size
        
        # 修复：使用买一和卖一价计算中间价
        if len(asks) > 0 and len(self.ex.fetch_order_book(asks[0][0])['bids']) > 0:
            mid_price = (float(asks[0][0]) + float(self.ex.fetch_order_book(asks[0][0])['bids'][0][0])) / 2
        else:
            mid_price = float(asks[0][0])
        
        dynamic_slippage = (avg_price - mid_price) / mid_price
        total_slippage = max(dynamic_slippage, base_slippage, base_slippage_config)
        return min(total_slippage, max_slippage_config)
    
    def _calculate_sell_slippage(self, bids, order_size):
        total_cost = 0
        filled_size = 0
        base_slippage = 0
        base_slippage_config = self.config_manager.get_config_value('trading_costs.base_slippage', 0.0005)
        max_slippage_config = self.config_manager.get_config_value('trading_costs.max_slippage', 0.01)
        
        for i, (price, volume) in enumerate(bids):
            price = float(price)
            volume = float(volume)
            if filled_size >= order_size:
                break
            available = min(volume, order_size - filled_size)
            total_cost += price * available
            filled_size += available
            if i >= 3:
                base_slippage += 0.0005
        
        if filled_size == 0:
            return max_slippage_config
        
        avg_price = total_cost / filled_size
        
        if len(bids) > 0 and len(self.ex.fetch_order_book(bids[0][0])['asks']) > 0:
            mid_price = (float(bids[0][0]) + float(self.ex.fetch_order_book(bids[0][0])['asks'][0][0])) / 2
        else:
            mid_price = float(bids[0][0])
        
        dynamic_slippage = (mid_price - avg_price) / mid_price
        total_slippage = max(dynamic_slippage, base_slippage, base_slippage_config)
        return min(total_slippage, max_slippage_config)

# ========= 增强的资金费率管理器 =========
class EnhancedFundingRateManager:
    """增强的资金费率管理器"""
    
    def __init__(self, exchange, config_manager):
        self.ex = exchange
        self.config_manager = config_manager
        self.funding_records = {}
        self.funding_cache = {}
    
    def should_avoid_trading(self, symbol, side, hours_before=None):
        try:
            if hours_before is None:
                hours_before = self.config_manager.get_config_value('realistic_trading.funding_avoid_hours', 4)
            
            fr, next_funding_time = self.get_funding_rate(symbol)
            current_time = datetime.now().timestamp()
            seconds_to_funding = (next_funding_time / 1000) - current_time
            hours_to_funding = seconds_to_funding / 3600
            
            if hours_to_funding < hours_before:
                LOG.warning(f'{symbol} {hours_to_funding:.1f}小时后资金费率结算，避免开仓')
                return True
            
            fee_threshold = self.config_manager.get_config_value('funding_limit', 0.001)  # 使用新的阈值
            
            if side == 'LONG' and fr > fee_threshold:
                LOG.warning(f'{symbol} 资金费率{fr:.4%}过高，不做多')
                return True
            elif side == 'SHORT' and fr < -fee_threshold:
                LOG.warning(f'{symbol} 资金费率{fr:.4%}过低，不做空')
                return True
            
            return False
        except Exception as e:
            LOG.error(f'资金费率检查失败: {e}')
            return False
    
    def get_funding_rate(self, symbol):
        """修复Gate.io资金费率获取"""
        cache_key = f"{symbol}_funding"
        cache_time = self.funding_cache.get(f"{cache_key}_time", 0)
        
        if time.time() - cache_time < 300 and cache_key in self.funding_cache:
            return self.funding_cache[cache_key]
        
        try:
            # Gate.io 资金费率API
            fr_data = retry(self.ex.fetch_funding_rate, symbol)
            
            # 适配Gate.io返回结构
            funding_rate = 0.0
            next_funding_time = 0
            
            if isinstance(fr_data, dict):
                funding_rate = float(fr_data.get('fundingRate', 0))
                next_funding_time = int(fr_data.get('nextFundingTime', 0))
            elif isinstance(fr_data, list) and len(fr_data) > 0:
                funding_rate = float(fr_data[0].get('fundingRate', 0))
                next_funding_time = int(fr_data[0].get('nextFundingTime', 0))
            
            # 验证数据有效性
            if abs(funding_rate) > 0.1:  # 异常值检查
                LOG.warning(f"{symbol} 资金费率异常: {funding_rate}, 使用保守值")
                funding_rate = 0.0001
            
            self.funding_cache[cache_key] = (funding_rate, next_funding_time)
            self.funding_cache[f"{cache_key}_time"] = time.time()
            
            return funding_rate, next_funding_time
            
        except Exception as e:
            LOG.error(f'获取资金费率失败: {e}')
            # 返回保守估计
            return 0.0001, int((datetime.now().timestamp() + 8 * 3600) * 1000)
    
    def calculate_funding_cost(self, symbol, position_size, entry_time, exit_time=None):
        try:
            if exit_time is None:
                exit_time = datetime.now()
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            if isinstance(exit_time, str):
                exit_time = datetime.fromisoformat(exit_time)
            
            holding_hours = (exit_time - entry_time).total_seconds() / 3600
            funding_cycles = int(holding_hours / 8)
            
            if funding_cycles < 1:
                return 0
            
            total_funding_cost = 0
            current_time = entry_time
            
            for cycle in range(funding_cycles):
                cycle_end = current_time + timedelta(hours=8)
                if cycle_end > exit_time:
                    cycle_end = exit_time
                
                fr, _ = self.get_funding_rate(symbol)
                cycle_hours = (cycle_end - current_time).total_seconds() / 3600
                cycle_cost = position_size * abs(fr) * (cycle_hours / 8)
                total_funding_cost += cycle_cost
                current_time = cycle_end
            
            return total_funding_cost
        except Exception as e:
            LOG.error(f'计算资金费用失败: {e}')
            return position_size * 0.0005
    
    def record_funding_payment(self, symbol, side, amount, timestamp):
        funding_record = {
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'timestamp': timestamp,
            'type': 'funding'
        }
        self.funding_records.setdefault(symbol, []).append(funding_record)

# ========= AI驱动的智能仓位管理器 =========
class AIPositionManager:
    """AI驱动的智能仓位管理"""
    
    def __init__(self, exchange, ai_optimizer, config_manager):
        self.ex = exchange
        self.ai_optimizer = ai_optimizer
        self.config_manager = config_manager
        self.performance_metrics = ai_optimizer.performance_metrics
        self.slippage_model = RealisticSlippageModel(exchange, config_manager)
    
    def get_ai_optimized_position_size(self, symbol, strategy_type, market_analysis):
        try:
            base_size = self._calculate_base_position(symbol, strategy_type, market_analysis)
            risk_adjusted = self._ai_risk_adjustment(base_size, symbol, strategy_type, market_analysis)
            market_adjusted = self._market_regime_adjustment(risk_adjusted, market_analysis)
            performance_adjusted = self._performance_feedback_adjustment(market_adjusted, symbol, strategy_type)
            final_size = self._final_risk_check(performance_adjusted, symbol)
            
            LOG.info(f"AI仓位决策: {symbol} {strategy_type} 基础{base_size:.0f} → 最终{final_size:.0f}")
            return final_size
        except Exception as e:
            LOG.error(f"AI仓位计算失败: {e}")
            return self._calculate_conservative_size()
    
    def _calculate_conservative_size(self):
        equity = self.ex.equity()
        return equity * self.config_manager.get_config_value('risk_management.base_risk_per_trade', 0.01)
    
    def _calculate_base_position(self, symbol, strategy_type, market_analysis):
        equity = self.ex.equity()
        base_risk = self._calculate_dynamic_base_risk(equity, market_analysis)
        
        strategy_multipliers = {
            'trend_breakout': 1.3,
            'trend_follow': 1.1,
            'grid_mean_reversion': 0.9,
            'grid_momentum': 1.0
        }
        strategy_multiplier = strategy_multipliers.get(strategy_type, 1.0)
        
        base_size = equity * base_risk * strategy_multiplier
        size_multiplier = self._get_capital_multiplier(equity)
        return base_size * size_multiplier
    
    def _calculate_dynamic_base_risk(self, equity, market_analysis):
        volatility_ratio = market_analysis.get('atr_ratio_50ma', 1.0)
        
        if volatility_ratio > 1.5:
            base_risk = 0.008
        elif volatility_ratio > 1.2:
            base_risk = 0.01
        elif volatility_ratio < 0.8:
            base_risk = 0.015
        else:
            base_risk = 0.012
        
        if equity > 100000:
            base_risk *= 0.8
        elif equity > 50000:
            base_risk *= 0.9
        
        return base_risk
    
    def _get_capital_tier(self, equity):
        for tier_name, tier_config in CFG['capital_tiers'].items():
            if tier_config['min'] <= equity < tier_config['max']:
                return tier_name
        return 'institutional'
    
    def _get_capital_multiplier(self, equity):
        if equity < 1000: return 1.0
        elif equity < 10000: return 1.2
        elif equity < 100000: return 1.5
        elif equity < 500000: return 1.8
        else: return 2.0
    
    def _ai_risk_adjustment(self, base_size, symbol, strategy_type, market_analysis):
        risk_score = self._ai_risk_assessment(symbol, strategy_type, market_analysis)
        adjustment_factors = {
            'very_low': 1.4,
            'low': 1.2,
            'medium': 1.0,
            'high': 0.7,
            'very_high': 0.5
        }
        return base_size * adjustment_factors.get(risk_score, 1.0)
    
    def _ai_risk_assessment(self, symbol, strategy_type, market_analysis):
        try:
            if not self.config_manager.get_config_value('ai_position_management.enabled', True):
                return 'medium'
            return self._simplified_risk_assessment(market_analysis)
        except Exception as e:
            LOG.error(f"AI风险评估失败: {e}")
            return 'medium'
    
    def _simplified_risk_assessment(self, market_analysis):
        risk_score = 0
        adx = market_analysis['adx']
        
        if adx > 40: risk_score += 2
        elif adx > 25: risk_score += 1
        elif adx < 15: risk_score -= 1
        
        atr_ratio = market_analysis['atr_ratio_50ma']
        if atr_ratio < 0.8: risk_score += 1
        elif atr_ratio > 1.5: risk_score -= 2
        
        rsi = market_analysis['rsi']
        if 30 < rsi < 70: risk_score += 1
        elif rsi > 85 or rsi < 15: risk_score -= 1
        
        if market_analysis['current_regime'] == 'trending': risk_score += 1
        else: risk_score -= 0.5
        
        if risk_score >= 3: return 'very_low'
        elif risk_score >= 1: return 'low'
        elif risk_score >= -1: return 'medium'
        elif risk_score >= -3: return 'high'
        else: return 'very_high'
    
    def _market_regime_adjustment(self, size, market_analysis):
        if not self.config_manager.get_config_value('ai_position_management.volatility_adjustment', True):
            return size
        
        regime = market_analysis['current_regime']
        volatility = market_analysis['volatility_regime']
        
        if regime == 'trending':
            size *= 1.2
        
        if volatility == 'high':
            size *= 0.7
        elif volatility == 'low':
            size *= 1.1
        
        return size
    
    def _performance_feedback_adjustment(self, size, symbol, strategy_type):
        if not self.config_manager.get_config_value('ai_position_management.performance_feedback', True):
            return size
        
        recent_performance = self._get_strategy_performance(symbol, strategy_type)
        if not recent_performance:
            return size
        
        win_rate = recent_performance.get('win_rate', 0.5)
        profit_factor = recent_performance.get('profit_factor', 1.0)
        
        adjustment = 1.0
        if win_rate > 0.6: adjustment *= 1.2
        elif win_rate < 0.4: adjustment *= 0.8
        
        if profit_factor > 1.5: adjustment *= 1.1
        elif profit_factor < 0.8: adjustment *= 0.9
        
        return size * adjustment
    
    def _get_strategy_performance(self, symbol, strategy_type):
        try:
            cutoff = datetime.now() - timedelta(days=30)
            recent_trades = [
                t for t in self.performance_metrics['trades']
                if (datetime.fromisoformat(t['timestamp']) > cutoff and
                    t['symbol'] == symbol and
                    t['regime'] == strategy_type)
            ]
            
            if len(recent_trades) < 5:
                return None
            
            pnls = [t['pnl'] for t in recent_trades]
            winning_trades = [p for p in pnls if p > 0]
            losing_trades = [p for p in pnls if p < 0]
            
            win_rate = len(winning_trades) / len(pnls) if pnls else 0
            avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(recent_trades)
            }
        except Exception as e:
            LOG.error(f"获取策略表现失败: {e}")
            return None
    
    def _final_risk_check(self, size, symbol):
        equity = self.ex.equity()
        
        # 单笔风险限制 - 使用放宽后的限制
        max_single_risk = equity * min(
            self.config_manager.get_config_value('ai_position_management.max_single_risk', 0.12),
            0.15  # 最高允许15%
        )
        size = min(size, max_single_risk)
        
        # 单个符号风险限制 - 使用放宽后的限制
        symbol_exposure = self._get_symbol_exposure(symbol)
        max_symbol_risk = equity * min(
            self.config_manager.get_config_value('ai_position_management.max_symbol_risk', 0.6),
            0.8  # 最高允许80%
        )
        available_symbol = max(0, max_symbol_risk - symbol_exposure)
        size = min(size, available_symbol)
        
        # 总风险限制 - 使用放宽后的限制
        total_exposure = self._get_total_exposure()
        max_total_risk = equity * min(
            self.config_manager.get_config_value('ai_position_management.max_total_risk', 1.2),
            1.5  # 最高允许150%
        )
        available_total = max(0, max_total_risk - total_exposure)
        size = min(size, available_total)
        
        # 最小名义金额 - 降低要求
        min_notional = self._get_dynamic_min_notional(equity)
        size = max(size, min_notional * 0.5)  # 降低到原来的50%
        
        return size
    
    def _get_symbol_exposure(self, symbol):
        try:
            positions = self.ex.ex.fetch_positions([symbol])
            total_notional = 0
            for pos in positions:
                if pos['symbol'] == symbol:
                    total_notional += abs(float(pos.get('notional', 0)))
            return total_notional
        except:
            return 0
    
    def _get_total_exposure(self):
        total = 0
        for symbol in CFG['symbols']:
            total += self._get_symbol_exposure(symbol)
        return total
    
    def _get_dynamic_min_notional(self, equity):
        """动态最小名义金额 - 降低门槛"""
        if equity < 500: return 5    # 降低门槛
        elif equity < 5000: return 20
        elif equity < 50000: return 50
        else: return 100

# ========= 异常处理增强 =========
class EnhancedExceptionHandler:
    """增强的异常处理"""
    
    CRITICAL_ERRORS = [
        'insufficient balance',
        'margin',
        'leverage',
        'permission',
        'authentication'
    ]
    
    @staticmethod
    def handle_trading_exception(e, symbol, context):
        error_str = str(e).lower()
        for critical in EnhancedExceptionHandler.CRITICAL_ERRORS:
            if critical in error_str:
                LOG.critical(f'🚨 关键错误 [{context}]: {e}')
                return 'CRITICAL'
        
        if isinstance(e, (ccxt.NetworkError, requests.exceptions.ConnectionError)):
            LOG.warning(f'🌐 网络错误 [{context}]: {e}')
            return 'NETWORK'
        
        if isinstance(e, ccxt.ExchangeError):
            if 'rate limit' in error_str:
                LOG.warning(f'⏰ 频率限制 [{context}]: {e}')
                return 'RATE_LIMIT'
            else:
                LOG.error(f'🏦 交易所错误 [{context}]: {e}')
                return 'EXCHANGE'
        
        LOG.error(f'⚠️ 业务错误 [{context}]: {e}')
        return 'BUSINESS'

# ========= 自动系统监控 =========
class AutoSystemMonitor:
    """全自动系统监控和恢复"""
    
    def __init__(self, trading_bot):
        self.bot = trading_bot
        self.health_check_interval = 300
        self.last_health_time = time.time()
        self.last_trade_activity = time.time()
        self.monitor_thread = None
        self.running = True
    
    def start_monitoring(self):
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        LOG.info("✅ 自动系统监控已启动")
    
    def _monitor_loop(self):
        while self.running:
            try:
                self._auto_health_check()
                self._auto_cleanup_old_files()
                time.sleep(60)
            except Exception as e:
                LOG.error(f'监控循环异常: {e}')
                time.sleep(30)
    
    def _auto_health_check(self):
        current_time = time.time()
        if current_time - self.last_health_time > self.health_check_interval:
            if not self._is_bot_active():
                LOG.warning("🔧 检测到系统可能暂停，执行自动恢复...")
                self._auto_recover()
            self.last_health_time = current_time
    
    def _is_bot_active(self):
        if time.time() - self.last_trade_activity < 600:
            return True
        try:
            self.bot.ex.ex.fetch_balance()
            return True
        except:
            return False
    
    def _auto_recover(self):
        LOG.info("🔄 执行自动恢复程序")
        try:
            self._cleanup_locks()
            self._reinitialize_components()
            self.bot.consecutive_errors = 0
            LOG.info("✅ 自动恢复完成")
        except Exception as e:
            LOG.error(f'自动恢复失败: {e}')
    
    def _cleanup_locks(self):
        lock_files = ['bot_gate.lock', 'ultimate_pos_ai_gate.json.lock']
        for lock_file in lock_files:
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                    LOG.info(f'已清理锁文件: {lock_file}')
            except:
                pass
    
    def _reinitialize_components(self):
        try:
            self.bot.pos = load_pos()
            self.bot.ex.circuit_breaker._initialized = False
            self.bot.ex.circuit_breaker.initialize()
            LOG.info("组件重新初始化完成")
        except Exception as e:
            LOG.error(f'组件重新初始化失败: {e}')
    
    def _auto_cleanup_old_files(self):
        try:
            current_time = time.time()
            for filename in os.listdir('.'):
                if filename.startswith('emergency_') and filename.endswith('.json'):
                    file_time = os.path.getctime(filename)
                    if current_time - file_time > 7 * 24 * 3600:
                        os.remove(filename)
                        LOG.info(f'已清理旧文件: {filename}')
        except Exception as e:
            LOG.debug(f'文件清理跳过: {e}')
    
    def stop_monitoring(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

# ========= 策略UID防重复 =========
def strategy_lock():
    """策略锁，支持Redis降级到文件锁"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True, socket_connect_timeout=2)
        code_hash = hashlib.md5(open(__file__, 'rb').read()).hexdigest()
        if r.get(code_hash):
            LOG.critical('🚨 同一策略已运行，退出')
            raise SystemExit
        r.setex(code_hash, 3600, 1)
        LOG.info('✅ Redis锁已设置')
        return r
    except Exception as e:
        LOG.warning(f'Redis不可用，使用文件锁: {e}')
        lock_file = 'bot_gate.lock'
        try:
            fd = open(lock_file, 'w')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            LOG.info('✅ 文件锁已设置')
            return fd
        except IOError:
            LOG.critical('🚨 同一策略已运行(文件锁)，退出')
            raise SystemExit

# 设置策略锁
lock_handler = strategy_lock()

# ========= 时钟同步 =========
def check_time_sync():
    try:
        # Gate.io 正确的时间同步接口
        response = requests.get('https://api.gateio.ws/api/v4/spot/time', timeout=5)
        if response.status_code == 200:
            server_time = response.json().get('server_time', 0)
        else:
            # 备用方法：使用交易所时间
            from ccxt import Gate
            ex = Gate()
            server_time = ex.fetch_time()
        
        local_time = int(time.time() * 1000)
        time_diff = abs(server_time - local_time)
        
        if time_diff > 5000:
            LOG.critical(f'🚨 本地时钟与交易所偏差>{time_diff}ms')
            raise SystemExit
        
        LOG.info(f'✅ 时钟同步正常，偏差: {time_diff}ms')
        return True
    except Exception as e:
        LOG.critical(f'🚨 时钟同步失败: {e}')
        raise SystemExit

def periodic_time_sync():
    """定期时钟同步"""
    retry_count = 0
    while True:
        time.sleep(3600)
        try:
            check_time_sync()
            retry_count = 0  # 重置重试计数
        except Exception as e:
            retry_count += 1
            LOG.warning(f"时钟同步失败第{retry_count}次: {e}")
            if retry_count >= 3:
                LOG.critical("🚨 时钟同步连续失败3次，停止交易")
                # 这里可以添加更严格的措施，如停止交易
            time.sleep(300)  # 等待5分钟后重试
            retry_count = 0

# 启动时钟同步线程
time_sync_thread = threading.Thread(target=periodic_time_sync, daemon=True)
time_sync_thread.start()

# ========= 网络退避 =========
def robust_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

SESSION = robust_session()

# ========= 工具函数 =========
def retry(func, *args, **kwargs):
    for i in range(3):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            LOG.warning(f'{func.__name__} err {i+1}: {e}')
            time.sleep(2 ** i)
    raise

def load_pos():
    """修复：使用文件锁加载仓位数据"""
    if not os.path.exists(CFG['position_file']):
        return {}
    
    try:
        with open(CFG['position_file'], 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return data
    except Exception as e:
        LOG.error(f'加载仓位数据失败: {e}')
        return {}

def save_pos(d):
    """修复：使用文件锁保存仓位数据"""
    try:
        with open(CFG['position_file'], 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(d, f, indent=2)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        LOG.error(f'保存仓位数据失败: {e}')

# ========= API权限自检 =========
def sanity_check_api(ex: ccxt.Exchange):
    try:
        # Gate.io 权限检查
        balance = ex.fetch_balance()
        if 'total' not in balance:
            LOG.critical('🚨 API密钥无合约权限'); return False
        
        # 尝试获取仓位信息
        positions = ex.fetch_positions([CFG['symbols'][0]])
        LOG.info('✅ API权限校验通过'); return True
    except Exception as e:
        LOG.critical(f'🚨 API权限异常: {e}'); return False

# ========= 未来函数防护 =========
def drop_last_bar(df: pd.DataFrame):
    return df[:-1]

# ========= quote币种验证 =========
def ensure_usdt_quote(ex: ccxt.Exchange, symbols):
    for s in symbols:
        try:
            market = ex.market(s)
            # Gate.io 永续合约符号格式检查
            if not (market['quote'] == 'USDT' or ':USDT' in s):
                LOG.critical(f'🚨 {s} 不是USDT本位')
                raise SystemExit
        except Exception as e:
            LOG.critical(f'🚨 验证交易对{s}失败: {e}')
            raise SystemExit

# ========= 交易所健康检查 =========
def exchange_health(ex: ccxt.Exchange):
    try:
        # Gate.io 健康检查
        ex.fetch_time()
        return True
    except:
        LOG.error('交易所维护/网络故障'); return False

# ========= 热重载 =========
def hot_reload_cfg():
    load_dotenv(override=True)
    CFG['max_equity_risk'] = float(os.getenv('MAX_RISK', CFG['max_equity_risk']))

# ========= 仓位操作锁 =========
position_lock = threading.RLock()

# ========= 增强熔断机制 (自动恢复版) =========
class AutoRecoveryCircuitBreaker:
    """增强熔断机制 - 自动恢复版本"""
    
    def __init__(self, exchange, config_manager):
        self.ex = exchange
        self.config_manager = config_manager
        self.max_drawdown = None
        self.daily_loss_limit = None
        self.position_level_breakers = {}
        self._initialized = False
        self._lock = threading.RLock()
        self.break_start_time = None
        self.auto_recovery_time = 3600
    
    def initialize(self):
        with self._lock:
            if not self._initialized:
                self.max_drawdown = self.config_manager.get_config_value('circuit_breaker.max_drawdown', 0.15)
                self.daily_loss_limit = self.config_manager.get_config_value('circuit_breaker.daily_loss_limit', 0.10)
                self.max_equity = self.ex.equity()
                today = datetime.now().date().isoformat()
                self.daily_equity = {today: self.ex.equity()}
                self._initialized = True
                LOG.info("熔断器初始化完成")
    
    def pre_trade_check(self, symbol, order_size, current_equity):
        if not self._initialized:
            self.initialize()
        
        with self._lock:
            if self.break_start_time is not None:
                if time.time() - self.break_start_time > self.auto_recovery_time:
                    LOG.info("🟢 熔断自动恢复，重置熔断状态")
                    self.break_start_time = None
                    # 修复：重新初始化权益数据
                    self.initialize()
                else:
                    return False, '熔断自动恢复期中'
            
            if not self.check_circuit():
                return False, '总资金熔断激活'
            
            # 放宽单笔风险检查
            risk_ratio = order_size / current_equity
            if risk_ratio > 0.15:  # 从 0.1 提高到 0.15
                return False, f'单笔风险过高: {risk_ratio:.1%}'
            
            # 放宽符号风险检查
            symbol_risk = self.get_symbol_exposure(symbol)
            if symbol_risk > 0.6:  # 从 0.4 提高到 0.6
                return False, f'符号风险过高: {symbol_risk:.1%}'
            
            # 放宽流动性检查
            if not self.liquidity_check(symbol, order_size):
                return False, '流动性不足'
            
            return True, '通过'
    
    def get_symbol_exposure(self, symbol):
        positions = self.ex.ex.fetch_positions([symbol])
        total_exposure = 0
        for pos in positions:
            if pos['symbol'] == symbol:
                total_exposure += abs(float(pos.get('notional', 0)))
        return total_exposure / self.ex.equity()
    
    def liquidity_check(self, symbol, order_size):
        try:
            orderbook = self.ex.fetch_order_book(symbol)
            bid_volume = sum([entry[1] for entry in orderbook['bids'][:3]])  # 只看前3档
            ask_volume = sum([entry[1] for entry in orderbook['asks'][:3]])
            min_volume = min(bid_volume, ask_volume)
            
            return order_size < min_volume * 0.2  # 从 0.1 提高到 0.2
        except:
            return True
    
    def check_circuit(self):
        if not self._initialized:
            self.initialize()
        
        with self._lock:
            current_max_drawdown = self.config_manager.get_config_value('circuit_breaker.max_drawdown', 0.15)
            current_daily_loss_limit = self.config_manager.get_config_value('circuit_breaker.daily_loss_limit', 0.10)
            
            if self.break_start_time is not None:
                time_elapsed = time.time() - self.break_start_time
                if time_elapsed > self.auto_recovery_time:
                    LOG.info("🟢 熔断自动恢复期结束，恢复正常交易")
                    self.break_start_time = None
                    # 修复：重新初始化权益数据
                    self.initialize()
                else:
                    remaining = self.auto_recovery_time - time_elapsed
                    LOG.info(f'⏳ 熔断自动恢复中，剩余时间: {remaining/60:.1f}分钟')
                    return False
            
            equity = self.ex.equity()
            
            # 修复：更新最大权益
            if equity > self.max_equity:
                self.max_equity = equity
            
            today = datetime.now().date().isoformat()
            if today not in self.daily_equity:
                self.daily_equity[today] = equity
            
            if self.max_equity > 0:
                drawdown = (self.max_equity - equity) / self.max_equity
                if drawdown > current_max_drawdown:
                    LOG.critical(f'🚨 回撤熔断 {drawdown:.2%}，自动恢复计时开始')
                    self.break_start_time = time.time()
                    return False
            
            daily_pnl = (equity - self.daily_equity[today]) / self.daily_equity[today]
            if daily_pnl < -current_daily_loss_limit:
                LOG.critical(f'🚨 日内亏损熔断 {daily_pnl:.2%}，自动恢复计时开始')
                self.break_start_time = time.time()
                return False
            
            return True

# ========= 交易所封装 (Gate.io版本) =========
class Exchange:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
        # Gate.io 交易所初始化
        self.ex = ccxt.gate({
            'apiKey': CFG['apiKey'],
            'secret': CFG['secret'],
            'sandbox': CFG['sandbox'],
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # Gate.io 永续合约类型
                'adjustForTimeDifference': True,
            },
            'session': SESSION
        })
        
        if not sanity_check_api(self.ex):
            raise RuntimeError('API自检失败')
        
        ensure_usdt_quote(self.ex, CFG['symbols'])
        
        # 设置杠杆和保证金模式
        for s in CFG['symbols']:
            try:
                # Gate.io 杠杆设置
                retry(self.ex.set_leverage, CFG['lev'], s)
                # Gate.io 保证金模式设置
                retry(self.ex.set_margin_mode, 'isolated', s)
            except Exception as e:
                LOG.warning(f"设置{s}杠杆/保证金模式失败: {e}，可能已经设置过")
        
        # 使用config_manager初始化各个组件
        self.circuit_breaker = AutoRecoveryCircuitBreaker(self, config_manager)
        self.slippage_model = RealisticSlippageModel(self, config_manager)
        self.funding_manager = EnhancedFundingRateManager(self, config_manager)
    
    def wait_filled(self, order, timeout=30):
        st = time.time()
        while time.time() - st < timeout:
            try:
                o = self.ex.fetch_order(order['id'], order['symbol'])
                if o['status'] == 'closed':
                    if float(o['filled']) == 0:
                        LOG.error('🚨 幽灵单：filled=0'); return None
                    return o
                elif o['status'] in ['canceled', 'expired', 'rejected', 'failed']:
                    LOG.warning(f'订单{order["id"]}状态: {o["status"]}')
                    filled = float(o.get('filled', 0))
                    if filled > 0:
                        LOG.info(f'订单部分成交: {filled}')
                        return o
                    return None
            except Exception as e:
                LOG.warning(f'查询订单状态失败: {e}')
            time.sleep(0.5)
        
        LOG.error(f'🚨 订单{order["id"]}未完全成交')
        try:
            o = self.ex.fetch_order(order['id'], order['symbol'])
            filled = float(o.get('filled', 0))
            if filled > 0:
                LOG.info(f'订单部分成交: {filled}')
                return o
        except:
            pass
        return None
    
    def place_market_order_realistic(self, symbol, side, amount_usdt, pos_side):
        if amount_usdt < self._get_min_notional():
            LOG.warning(f'{symbol} 名义<{amount_usdt} 跳过')
            return None
        
        # 滑点模型计算
        slippage = self.slippage_model.calculate_dynamic_slippage(symbol, amount_usdt, side)
        ticker = retry(self.ex.fetch_ticker, symbol)
        mid_price = float(ticker['last'])
        
        # 应用滑点
        if side == 'buy':
            execution_price = mid_price * (1 + slippage)
        else:
            execution_price = mid_price * (1 - slippage)
        
        LOG.info(f'{symbol} {side} 滑点: {slippage:.3%}, 执行价: {execution_price:.4f}')
        
        amount = PrecisionHelper.safe_float_operation(amount_usdt, execution_price, symbol, self.ex)
        if amount is None:
            return None
        
        try:
            # Gate.io 市价单参数
            order_params = {
                'timeInForce': 'IOC',
                'positionSide': pos_side  # 添加仓位方向
            }
            
            # 验证仓位方向有效性
            if pos_side not in ['LONG', 'SHORT']:
                LOG.error(f"无效的仓位方向: {pos_side}")
                return None
            
            # 使用限价单模拟真实执行
            order = self.ex.create_order(
                symbol, 'limit', side, amount, execution_price, order_params
            )
            return self.wait_filled(order)
        except Exception as e:
            LOG.error(f'真实滑点订单失败: {e}')
            # 备用：尝试直接市价单
            try:
                LOG.info(f'尝试使用市价单: {symbol} {side} {amount}')
                market_order = self.ex.create_order(
                    symbol, 'market', side, amount, None, {'positionSide': pos_side}
                )
                return self.wait_filled(market_order)
            except Exception as market_e:
                LOG.error(f'市价单也失败: {market_e}')
                return None
    
    def calculate_total_trading_costs(self, symbol, side, trade_value, is_opening=True):
        fee_taker = self.config_manager.get_config_value('trading_costs.taker_fee', 0.0005)
        fee_maker = self.config_manager.get_config_value('trading_costs.maker_fee', 0.0002)
        base_slippage = self.config_manager.get_config_value('trading_costs.base_slippage', 0.0005)
        
        fee_rate = fee_taker if is_opening else fee_maker
        fee_cost = trade_value * fee_rate
        
        liquidity_adjustment = self._get_liquidity_adjustment(symbol, trade_value)
        adjusted_slippage = base_slippage * liquidity_adjustment
        slippage_cost = trade_value * adjusted_slippage
        
        total_cost = fee_cost + slippage_cost
        LOG.debug(f'{symbol} {side} 交易成本: 手续费{fee_cost:.2f} + 滑点{slippage_cost:.2f} = 总计{total_cost:.2f}')
        return total_cost
    
    def _get_liquidity_adjustment(self, symbol, trade_value):
        try:
            orderbook = self.ex.fetch_order_book(symbol, limit=10)
            bid_volume = sum([entry[1] for entry in orderbook['bids'][:5]])
            ask_volume = sum([entry[1] for entry in orderbook['asks'][:5]])
            avg_volume = (bid_volume + ask_volume) / 2
            
            if avg_volume == 0:
                return 2.0
            
            volume_ratio = trade_value / avg_volume
            
            if volume_ratio > 0.1:
                return 3.0
            elif volume_ratio > 0.05:
                return 2.0
            elif volume_ratio > 0.02:
                return 1.5
            else:
                return 1.0
        except Exception as e:
            LOG.warning(f'流动性评估失败: {e}')
            return 1.5
    
    def gen_cid(self, symbol, side):
        return f"GateAIv2_{symbol.replace('/', '').replace(':', '')}_{side}_{int(time.time()*1000)}"
    
    def fetch_mtf(self, symbol):
        mtf = {}
        for tf in CFG['timeframes']:
            try:
                # 确保使用正确的符号格式
                ohlcv = retry(self.ex.fetch_ohlcv, symbol, tf, limit=150)  # 减少数据量
                if not ohlcv or len(ohlcv) == 0:
                    LOG.warning(f'获取{symbol} {tf} K线数据为空')
                    continue
                
                df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                
                if not self.validate_ohlcv_data(df, symbol, tf):
                    continue
                
                df = drop_last_bar(df)
                
                df['tr'] = pd.concat([(df['h'] - df['l']), (df['h'] - df['c'].shift(1)).abs(),
                                    (df['l'] - df['c'].shift(1)).abs()], axis=1).max(axis=1)
                df['atr'] = df['tr'].rolling(10).mean()  # 从 14 减少到 10
                
                df['bb_mid'] = df['c'].rolling(15).mean()  # 从 20 减少到 15
                std = df['c'].rolling(15).std()
                df['bb_up'] = df['bb_mid'] + 1.8 * std    # 从 2.0 降低到 1.8
                df['bb_low'] = df['bb_mid'] - 1.8 * std
                df['bb_width'] = (df['bb_up'] - df['bb_low']) / df['bb_mid']
                
                df['adx'], df['plus_di'], df['minus_di'] = self._adx(df, 10)  # 周期从14减少到10
                
                df['rsi'] = self._rsi(df, 10)  # RSI周期从14减少到10
                
                # 放宽趋势判断条件
                adx_val = df['adx'].iloc[-1] if not pd.isna(df['adx'].iloc[-1]) else 0
                avg_w = df['bb_width'].rolling(30).mean().iloc[-1]  # 从50减少到30
                
                # 动态趋势阈值
                dynamic_threshold = 18  # 从25降低到18
                df['regime'] = 'trending' if (adx_val > dynamic_threshold and 
                                            df['bb_width'].iloc[-1] > avg_w * 0.8) else 'ranging'  # 从 > avg_w 降低到 > 0.8 * avg_w
                
                mtf[tf] = df
                
            except Exception as e:
                LOG.error(f'获取{symbol} {tf} K线数据失败: {e}')
                continue
        return mtf
    
    def validate_ohlcv_data(self, df, symbol, timeframe):
        """增强数据验证"""
        try:
            # 检查空值
            if df.isnull().any().any():
                LOG.warning(f"{symbol} {timeframe} 数据包含空值")
                return False
            
            # 检查数据量
            if len(df) < 30:
                LOG.warning(f"{symbol} {timeframe} 数据量不足: {len(df)}")
                return False
            
            # 检查价格合理性
            if (df['h'] < df['l']).any() or (df['h'] < df['c']).any() or (df['l'] > df['c']).any():
                LOG.warning(f"{symbol} {timeframe} 价格数据异常")
                return False
            
            # 检查价格是否为正
            if (df[['o', 'h', 'l', 'c']] <= 0).any().any():
                LOG.warning(f"{symbol} {timeframe} 价格非正")
                return False
            
            # 检查成交量是否为负
            if (df['v'] < 0).any():
                LOG.warning(f"{symbol} {timeframe} 成交量为负")
                return False
            
            # 时间连续性检查
            time_diff = df['ts'].diff().dt.total_seconds()
            expected_interval = {
                '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400
            }.get(timeframe, 900)
            
            if (time_diff.iloc[1:] > expected_interval * 2).any():
                LOG.warning(f"{symbol} {timeframe} 时间连续性异常")
                return False
            
            return True
        except Exception as e:
            LOG.error(f"数据验证失败: {e}")
            return False
    
    @staticmethod
    def _adx(df, p=10):  # 周期从14减少到10
        high, low, close = df['h'], df['l'], df['c']
        tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(p).mean()
        
        up, down = high - high.shift(1), low.shift(1) - low
        plus_dm = (up.where((up > down) & (up > 0), 0)).rolling(p).mean()
        minus_dm = (down.where((down > up) & (down > 0), 0)).rolling(p).mean()
        
        plus_di, minus_di = 100 * (plus_dm / atr), 100 * (minus_dm / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(p).mean()
        return adx, plus_di, minus_di
    
    @staticmethod
    def _rsi(df, p=10):  # 周期从14减少到10
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def equity(self):
        try:
            balance = retry(self.ex.fetch_balance)
            # Gate.io 正确的余额结构
            if 'total' in balance:
                return float(balance['total'].get('USDT', 0))
            elif 'info' in balance and 'total' in balance['info']:
                return float(balance['info']['total'].get('USDT', 0))
            else:
                # 备用方法
                for currency, info in balance.items():
                    if currency == 'USDT' and 'total' in info:
                        return float(info['total'])
                return 1000  # 保守默认值
        except Exception as e:
            LOG.error(f'获取权益失败: {e}')
            return 1000
    
    def fetch_positions(self, symbol):
        try:
            # Gate.io 仓位查询方式
            positions = retry(self.ex.fetch_positions, [symbol])
            long_p = short_p = None
            for pos in positions:
                # Gate.io 仓位数据结构适配
                if pos['symbol'] == symbol or pos['symbol'].replace('_', '/') == symbol.replace('_', '/'):
                    # 根据持仓方向分类
                    if float(pos.get('contracts', 0)) > 0:
                        if pos.get('side') == 'long' or (pos.get('side') is None and float(pos.get('notional', 0)) > 0):
                            long_p = pos
                    elif pos.get('side') == 'short' or (pos.get('side') is None and float(pos.get('notional', 0)) < 0):
                        short_p = pos
            return long_p, short_p
        except Exception as e:
            LOG.error(f'获取{symbol}仓位失败: {e}')
            return None, None
    
    def fetch_funding_rate(self, symbol):
        try:
            fr_data = retry(self.ex.fetch_funding_rate, symbol)
            funding_rate = fr_data.get('fundingRate', 0)
            next_funding_time = fr_data.get('nextFundingTime', 0)
            return funding_rate, next_funding_time
        except Exception as e:
            LOG.error(f"获取资金费率失败: {e}")
            return 0.001, 0
    
    def fetch_risk_limit(self, symbol):
        m = self.ex.market(symbol)
        return m['limits']['leverage']['max'], m['limits']['amount']['max']
    
    def calculate_liquidation_price(self, symbol, side, entry_price, leverage, margin_mode='isolated'):
        try:
            if margin_mode == 'isolated':
                if side == 'LONG':
                    return entry_price * (1 - 1/leverage + 0.004)
                else:
                    return entry_price * (1 + 1/leverage - 0.004)
            else:
                return entry_price * (1 - 0.9/leverage) if side == 'LONG' else entry_price * (1 + 0.9/leverage)
        except:
            return entry_price * 0.9 if side == 'LONG' else entry_price * 1.1
    
    def mm_buffer(self, symbol, side, price, atr, position_size):
        try:
            market = self.ex.market(symbol)
            contract_size = float(market['contractSize'])
            mm_rate = float(market.get('maintenanceMarginRate', 0.02))
            current_leverage = self.config_manager.get_config_value('lev', 15)
            
            liq_price = self.calculate_liquidation_price(symbol, side, price, current_leverage)
            stop_atr_multiplier = 2.0
            stop_price = price - stop_atr_multiplier * atr if side == 'LONG' else price + stop_atr_multiplier * atr
            
            buffer_distance = atr * 0.5
            
            if side == 'LONG':
                if stop_price >= liq_price - buffer_distance:
                    LOG.error(f'🚨 止损价{stop_price:.2f}太靠近强平价{liq_price:.2f}')
                    return False
            else:
                if stop_price <= liq_price + buffer_distance:
                    LOG.error(f'🚨 止损价{stop_price:.2f}太靠近强平价{liq_price:.2f}')
                    return False
            
            return True
        except Exception as e:
            LOG.error(f'维持保证金检查失败: {e}')
            return False
    
    def close_all_positions(self, symbol, side, pos_side, units, unit_usdt):
        if units <= 0: return True
        
        amount_usdt = units * unit_usdt
        side2 = 'sell' if side == 'LONG' else 'buy'
        
        with position_lock:
            check_passed, reason = self.circuit_breaker.pre_trade_check(symbol, amount_usdt, self.equity())
            if not check_passed:
                LOG.warning(f'平仓被熔断阻止: {reason}')
                return False
            
            order = self.place_market_order_realistic(symbol, side2, amount_usdt, pos_side)
            if not order: return False
            
            max_wait = 30; st = time.time()
            while time.time() - st < max_wait:
                long_p, short_p = self.fetch_positions(symbol)
                current_pos = long_p if side == 'LONG' else short_p
                if current_pos is None:
                    remain = 0
                else:
                    remain = abs(float(current_pos.get('positionAmt', 0)))
                
                if remain < 0.001:
                    LOG.info(f'{symbol} {side} 平仓完成'); return True
                time.sleep(1)
            
            LOG.error(f'{symbol} {side} 平仓超时'); return False
    
    def place_market_order(self, symbol, side, amount_usdt, position_side):
        if amount_usdt < self._get_min_notional():
            LOG.warning(f'{symbol} 名义<{amount_usdt} 跳过'); return None
        
        check_passed, reason = self.circuit_breaker.pre_trade_check(symbol, amount_usdt, self.equity())
        if not check_passed:
            LOG.warning(f'下单被熔断阻止: {reason}')
            return None
        
        return self.place_market_order_realistic(symbol, side, amount_usdt, position_side)
    
    def _get_min_notional(self):
        equity = self.equity()
        if equity < 500: return 5    # 降低门槛
        elif equity < 5000: return 20
        elif equity < 50000: return 50
        else: return 100
    
    def get_contract_value(self, symbol):
        try:
            market = self.ex.market(symbol)
            return float(market['contractSize'])
        except:
            return 1.0
    
    def fetch_open_orders(self, symbol=None):
        """获取未成交订单"""
        try:
            if symbol:
                return retry(self.ex.fetch_open_orders, symbol)
            else:
                return retry(self.ex.fetch_open_orders)
        except Exception as e:
            LOG.error(f"获取未成交订单失败: {e}")
            return []
    
    def fetch_order(self, order_id, symbol):
        """获取订单详情"""
        try:
            return retry(self.ex.fetch_order, order_id, symbol)
        except Exception as e:
            LOG.error(f"获取订单{order_id}详情失败: {e}")
            return None
    
    def cancel_order(self, order_id, symbol):
        """取消订单"""
        try:
            return retry(self.ex.cancel_order, order_id, symbol)
        except Exception as e:
            LOG.error(f"取消订单{order_id}失败: {e}")
            return None

# ========= 内存安全性能指标管理 =========
class MemorySafePerformanceMetrics:
    """增强内存安全管理"""
    
    def __init__(self):
        self.max_trades = 1000
        self.max_regimes = 5000
        self.cleanup_interval = 50  # 降低清理间隔
        self.operation_count = 0
        self.last_cleanup_time = time.time()
    
    def load_performance_metrics(self):
        if not os.path.exists(CFG['performance_file']):
            return self._get_default_metrics()
        
        try:
            with open(CFG['performance_file'], 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                d = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            trades = d.get('trades', [])[-self.max_trades:]
            regimes = d.get('market_regimes', [])[-self.max_regimes:]
            
            return {
                'trades': deque(trades, maxlen=self.max_trades),
                'market_regimes': deque(regimes, maxlen=self.max_regimes),
                'daily_equity': d.get('daily_equity', []),
                'parameter_history': d.get('parameter_history', [])
            }
        except Exception as e:
            LOG.error(f'加载性能数据失败: {e}')
            return self._get_default_metrics()
    
    def _get_default_metrics(self):
        return {
            'trades': deque(maxlen=self.max_trades),
            'market_regimes': deque(maxlen=self.max_regimes),
            'daily_equity': [],
            'parameter_history': []
        }
    
    def save_performance_metrics(self, metrics):
        try:
            d = {k: list(v) if isinstance(v, deque) else v for k, v in metrics.items()}
            with open(CFG['performance_file'], 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(d, f, indent=2)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            LOG.error(f'保存性能数据失败: {e}')
    
    def record_trade(self, metrics, trade_data):
        """记录交易 - 增强内存管理"""
        try:
            # 定期强制清理
            current_time = time.time()
            if (current_time - self.last_cleanup_time > 3600 or 
                self.operation_count >= self.cleanup_interval):
                self.force_cleanup(metrics)
                self.last_cleanup_time = current_time
                self.operation_count = 0
            
            # 检查deque长度
            if len(metrics['trades']) >= self.max_trades:
                removed = metrics['trades'].popleft()
                LOG.debug(f"清理最旧交易记录: {removed.get('timestamp', 'unknown')}")
            
            metrics['trades'].append(trade_data)
            self.operation_count += 1
            
        except Exception as e:
            LOG.error(f"记录交易失败: {e}")
            # 紧急清理
            try:
                self.force_cleanup(metrics)
            except:
                pass
    
    def force_cleanup(self, metrics):
        """强制清理内存"""
        try:
            # 清理交易记录
            if len(metrics['trades']) > int(self.max_trades * 0.8):
                keep_count = int(self.max_trades * 0.7)
                metrics['trades'] = deque(
                    list(metrics['trades'])[-keep_count:],
                    maxlen=self.max_trades
                )
                LOG.info(f"交易记录清理至{keep_count}条")
            
            # 清理市场状态记录
            if len(metrics['market_regimes']) > int(self.max_regimes * 0.8):
                keep_count = int(self.max_regimes * 0.7)
                metrics['market_regimes'] = deque(
                    list(metrics['market_regimes'])[-keep_count:],
                    maxlen=self.max_regimes
                )
                LOG.info(f"市场状态记录清理至{keep_count}条")
            
            # 触发垃圾回收
            gc.collect()
            LOG.debug("内存清理完成")
            
        except Exception as e:
            LOG.error(f"内存清理失败: {e}")

# ========= 安全的AI优化器 (自动恢复版) =========
class SafeAIOptimizer:
    """安全的AI参数优化器 - 自动恢复版本"""
    
    def __init__(self, config_manager):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.last_optimization_time = 0
        self.memory_manager = MemorySafePerformanceMetrics()
        self.performance_metrics = self.memory_manager.load_performance_metrics()
        self.config_manager = config_manager
        self.safety_limits = {
            'base_trend_unit': (50, 300),
            'base_grid_unit': (20, 150),
            'trend_stop_atr': (1.0, 4.0),
            'grid_stop_atr': (0.5, 3.0),
            'grid_take_profit_atr': (1.0, 4.0),
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'adx_trend_threshold': (15, 35),
            'max_position_ratio': (0.5, 3.0)
        }
    
    def save_performance_metrics(self):
        self.memory_manager.save_performance_metrics(self.performance_metrics)
    
    def record_trade(self, symbol, side, entry, exit_price, qty, pnl, regime, params):
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'entry_price': entry,
            'exit_price': exit_price,
            'quantity': qty,
            'pnl': pnl,
            'regime': regime,
            'parameters': params
        }
        self.memory_manager.record_trade(self.performance_metrics, trade)
        self.save_performance_metrics()
    
    def record_market_regime(self, symbol, regime, metrics):
        self.performance_metrics['market_regimes'].append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'regime': regime,
            'metrics': metrics
        })
    
    def comprehensive_backtest(self, df: pd.DataFrame, params: dict, initial_capital=10000):
        if len(df) < 200:
            return {'score': 0, 'reason': '数据不足'}
        
        try:
            results = {}
            for lookback in [30, 60, 90]:
                if len(df) >= lookback:
                    test_df = df.iloc[-lookback:]
                    results[lookback] = self.single_period_backtest(test_df, params, initial_capital)
            
            consistency_score = self.check_consistency(results)
            stress_result = self.stress_test(df, params)
            final_score = self.calculate_comprehensive_score(results, stress_result, consistency_score)
            
            return {
                'score': final_score,
                'details': results,
                'stress_test': stress_result,
                'consistency': consistency_score
            }
        except Exception as e:
            LOG.error(f'回测失败: {e}')
            return {'score': 0, 'reason': str(e)}
    
    def stress_test(self, df, params):
        volatility = df['atr'] / df['c']
        high_vol_periods = volatility.nlargest(5).index
        stress_scores = []
        
        for period in high_vol_periods:
            idx = df.index.get_loc(period)
            if 50 <= idx <= len(df) - 50:
                stress_data = df.iloc[idx-50:idx+50]
                score = self.single_period_backtest(stress_data, params, 10000)
                stress_scores.append(score.get('sharpe', 0))
        
        return {
            'avg_stress_sharpe': np.mean(stress_scores) if stress_scores else 0,
            'min_stress_sharpe': min(stress_scores) if stress_scores else 0,
            'passed': len([s for s in stress_scores if s > -1]) >= 3
        }
    
    def single_period_backtest(self, df: pd.DataFrame, params: dict, initial_capital=10000):
        try:
            if len(df) < 100:
                return {'sharpe': 0, 'max_drawdown': 1, 'win_rate': 0}
            
            fee_taker = self.config_manager.get_config_value('trading_costs.taker_fee', 0.0005)
            fee_maker = self.config_manager.get_config_value('trading_costs.maker_fee', 0.0002)
            base_slippage = self.config_manager.get_config_value('trading_costs.base_slippage', 0.0005)
            funding_rate = 0.0001
            
            signals = []
            positions = []
            entry_prices = []
            entry_times = []
            equity_curve = [initial_capital]
            current_equity = initial_capital
            
            for i in range(50, len(df)-1):
                row = df.iloc[i]
                next_row = df.iloc[i+1]
                prev_data = df.iloc[:i+1]
                
                if len(prev_data) > 20:
                    adx_val = prev_data['adx'].iloc[-1] if not pd.isna(prev_data['adx'].iloc[-1]) else 0
                    rsi_val = prev_data['rsi'].iloc[-1]
                    
                    if adx_val > params.get('adx_trend_threshold', 25):
                        if rsi_val > 50:
                            signal = 1
                        else:
                            signal = -1
                    else:
                        signal = 0
                else:
                    signal = 0
                
                signals.append(signal)
                
                atr = row['atr'] if not pd.isna(row['atr']) else row['tr'] * 0.01
                risk_per_trade = current_equity * self.config_manager.get_config_value('risk_management.base_risk_per_trade', 0.01)
                contract_size = 1.0
                
                if atr > 0 and contract_size > 0:
                    dynamic_units = risk_per_trade / (atr * contract_size)
                else:
                    dynamic_units = risk_per_trade / 100
                
                max_position_ratio = params.get('max_position_ratio', 2.5)
                dynamic_units = min(dynamic_units, current_equity * max_position_ratio / row['c'])
                
                if positions and positions[-1] != 0:
                    prev_position = positions[-1]
                    entry_price = entry_prices[-1]
                    entry_time = entry_times[-1]
                    
                    if prev_position == 1:
                        exit_price = next_row['l'] * (1 - base_slippage)
                    else:
                        exit_price = next_row['h'] * (1 + base_slippage)
                    
                    trade_value = abs(dynamic_units) * entry_price
                    total_fee = trade_value * (fee_taker + fee_maker)
                    slippage_cost = trade_value * base_slippage
                    
                    holding_hours = (i - entry_time) * 1
                    funding_cycles = holding_hours // 8
                    funding_cost = trade_value * funding_rate * funding_cycles
                    
                    if prev_position == 1:
                        price_pnl = (exit_price - entry_price) * dynamic_units
                    else:
                        price_pnl = (entry_price - exit_price) * dynamic_units
                    
                    net_pnl = price_pnl - total_fee - slippage_cost - funding_cost
                    pnl_ratio = net_pnl / current_equity
                    new_equity = current_equity + net_pnl
                    
                    if signal != prev_position:
                        equity_curve.append(new_equity)
                        current_equity = new_equity
                        positions.append(0)
                        entry_prices.append(0)
                        entry_times.append(0)
                    else:
                        equity_curve.append(current_equity)
                        positions.append(prev_position)
                        entry_prices.append(entry_price)
                        entry_times.append(entry_time)
                else:
                    if signal != 0:
                        if signal == 1:
                            execution_price = next_row['h'] * (1 + base_slippage)
                        else:
                            execution_price = next_row['l'] * (1 - base_slippage)
                        
                        positions.append(signal)
                        entry_prices.append(execution_price)
                        entry_times.append(i)
                        equity_curve.append(current_equity)
                    else:
                        positions.append(0)
                        entry_prices.append(0)
                        entry_times.append(0)
                        equity_curve.append(current_equity)
            
            returns = []
            for i in range(1, len(equity_curve)):
                if equity_curve[i-1] > 0:
                    ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                    returns.append(ret)
            
            drawdowns = []
            peak_equity = initial_capital
            for equity in equity_curve:
                if equity > peak_equity:
                    peak_equity = equity
                drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                drawdowns.append(drawdown)
            
            if len(returns) < 10:
                return {'sharpe': 0, 'max_drawdown': 1, 'win_rate': 0}
            
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24 * 4) if np.std(returns) > 0 else 0
            max_dd = max(drawdowns) if drawdowns else 0
            win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
            
            positive_returns = [r for r in returns if r > 0]
            negative_returns = [r for r in returns if r < 0]
            
            if negative_returns and positive_returns:
                profit_factor = abs(sum(positive_returns) / sum(negative_returns))
            else:
                profit_factor = float('inf') if positive_returns else 0
            
            total_return = equity_curve[-1] / initial_capital - 1 if equity_curve else 0
            
            return {
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_return': total_return,
                'final_equity': equity_curve[-1] if equity_curve else initial_capital
            }
        except Exception as e:
            LOG.error(f'单周期回测失败: {e}')
            return {'sharpe': 0, 'max_drawdown': 1, 'win_rate': 0}
    
    def check_consistency(self, results):
        if not results:
            return 0
        
        sharpe_scores = [r.get('sharpe', 0) for r in results.values()]
        if np.mean(sharpe_scores) == 0:
            return 1
        
        return np.std(sharpe_scores) / (np.mean(sharpe_scores) + 1e-8)
    
    def calculate_comprehensive_score(self, results, stress_result, consistency_score):
        if not results:
            return 0
        
        avg_sharpe = np.mean([r.get('sharpe', 0) for r in results.values()])
        avg_drawdown = np.mean([r.get('max_drawdown', 1) for r in results.values()])
        drawdown_penalty = max(0, 1 - avg_drawdown * 5)
        stress_score = 1 if stress_result.get('passed', False) else 0.5
        consistency = max(0, 1 - consistency_score * 2)
        
        final_score = avg_sharpe * 0.4 + drawdown_penalty * 0.3 + stress_score * 0.2 + consistency * 0.1
        return max(0, final_score)
    
    def calculate_performance_metrics(self, lookback_days=30):
        if not self.performance_metrics['trades']:
            return None
        
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent = [t for t in self.performance_metrics['trades']
                  if datetime.fromisoformat(t['timestamp']) > cutoff]
        
        if len(recent) < self.config_manager.get_config_value('ai_optimization.min_trades_for_optimization', 8):
            return None
        
        pnls = [t['pnl'] for t in recent]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        total_pnl = sum(pnls)
        win_rate = len(winning_trades) / len(pnls) if pnls else 0
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else float('inf')
        
        trending_trades = [t for t in recent if t['regime'] == 'trending']
        ranging_trades = [t for t in recent if t['regime'] == 'ranging']
        
        equity_curve = [10000]
        for pnl in pnls:
            equity_curve.append(equity_curve[-1] + pnl)
        
        peak = equity_curve[0]
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'total_trades': len(recent),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trending_trades': len(trending_trades),
            'trending_pnl': sum(t['pnl'] for t in trending_trades),
            'ranging_trades': len(ranging_trades),
            'ranging_pnl': sum(t['pnl'] for t in ranging_trades),
            'max_drawdown': max_drawdown,
            'lookback_days': lookback_days
        }
    
    def should_optimize_dynamic(self, market_analysis):
        hours_since_last = (time.time() - self.last_optimization_time) / 3600
        base_interval = self.config_manager.get_config_value('ai_optimization.base_interval_hours', 12)
        
        if hours_since_last < base_interval:
            return False, f"时间未到: {hours_since_last:.1f}/{base_interval}小时"
        
        pm = self.calculate_performance_metrics(self.config_manager.get_config_value('ai_optimization.backtest_lookback_days', 30))
        if not pm:
            return False, "性能数据不足"
        
        if pm['win_rate'] < 0.35 and pm['profit_factor'] < 1.1:
            return True, f"性能恶化: 胜率{pm['win_rate']:.1%}, 盈亏比{pm['profit_factor']:.2f}"
        
        if 'df' in market_analysis:
            df = market_analysis['df']
            if len(df) > 50:
                recent_vol = df['atr'].iloc[-20:].mean() / df['c'].iloc[-20:].mean()
                if recent_vol > 0.04:
                    dynamic_interval = self.config_manager.get_config_value('ai_optimization.high_frequency_interval', 6)
                else:
                    dynamic_interval = self.config_manager.get_config_value('ai_optimization.low_frequency_interval', 24)
                
                if hours_since_last >= dynamic_interval:
                    return True, f"波动性触发: {dynamic_interval}小时"
        
        if hours_since_last >= base_interval:
            return True, f"定期优化: {base_interval}小时"
        
        return False, f"等待优化: {hours_since_last:.1f}小时"
    
    def market_regime_shift(self, market_data):
        try:
            if 'df' not in market_data:
                return False
            
            df = market_data['df']
            if len(df) < 100:
                return False
            
            recent_adx = df['adx'].iloc[-50:].mean()
            curr_adx = df['adx'].iloc[-1]
            adx_shift = abs(curr_adx - recent_adx) / recent_adx if recent_adx > 0 else 0
            
            recent_vol = df['atr'].iloc[-50:].mean()
            curr_vol = df['atr'].iloc[-1]
            vol_shift = abs(curr_vol - recent_vol) / recent_vol if recent_vol > 0 else 0
            
            return adx_shift > 0.4 and vol_shift > 0.3
        except:
            return False
    
    def safe_optimization(self, current_params, performance_metrics, market_analysis):
        if not self.pre_optimization_checks():
            return current_params
        
        ai_params = self.get_ai_optimization(current_params, performance_metrics, market_analysis)
        if not ai_params:
            return current_params
        
        safe_params = self.validate_ai_safety(ai_params, current_params)
        if not safe_params:
            return current_params
        
        risk_adjusted = self.adjust_for_risk(safe_params, market_analysis)
        
        if self.final_validation(risk_adjusted, current_params, market_analysis):
            return risk_adjusted
        else:
            return current_params
    
    def pre_optimization_checks(self):
        if not self.api_key or self.api_key == 'your_deepseek_key':
            LOG.warning('AI API密钥未配置，跳过优化')
            return False
        
        if not self.config_manager.get_config_value('ai_optimization.enabled', True):
            return False
        
        return True
    
    def validate_ai_safety(self, new_params, current_params):
        validated = current_params.copy()
        
        for key, (min_val, max_val) in self.safety_limits.items():
            if key in new_params:
                current_val = current_params.get(key, min_val)
                if abs(current_val) < 1e-8:
                    max_change = min_val * 0.3
                else:
                    max_change = abs(current_val) * 0.3
                
                safe_value = np.clip(
                    new_params[key],
                    max(min_val, current_val - max_change),
                    min(max_val, current_val + max_change)
                )
                validated[key] = safe_value
                
                if abs(new_params[key] - safe_value) > 1e-8:
                    LOG.warning(f'AI参数{key}从{new_params[key]}调整为{safe_value}')
        
        return validated
    
    def adjust_for_risk(self, params, market_analysis):
        if market_analysis.get('volatility_regime') == 'high':
            params['base_trend_unit'] *= 0.7
            params['base_grid_unit'] *= 0.7
            params['max_position_ratio'] = min(params['max_position_ratio'] * 0.8, 1.5)
        
        if market_analysis.get('current_regime') == 'trending':
            params['trend_stop_atr'] = min(params['trend_stop_atr'], 2.5)
        else:
            params['grid_stop_atr'] = min(params['grid_stop_atr'], 1.5)
        
        return params
    
    def final_validation(self, new_params, current_params, market_analysis):
        if 'df' not in market_analysis:
            return True
        
        backtest_result = self.comprehensive_backtest(market_analysis['df'], new_params)
        current_result = self.comprehensive_backtest(market_analysis['df'], current_params)
        
        new_score = backtest_result.get('score', 0)
        current_score = current_result.get('score', 0)
        
        if new_score < current_score * self.config_manager.get_config_value('ai_optimization.validation_threshold', 0.8):
            LOG.warning(f'AI参数回测退化: {new_score:.3f} < {current_score:.3f}，放弃')
            return False
        
        return True
    
    def optimize_parameters(self, current_params, market_analysis):
        should_optimize, reason = self.should_optimize_dynamic(market_analysis)
        if not should_optimize:
            return current_params
        
        pm = self.calculate_performance_metrics(self.config_manager.get_config_value('ai_optimization.backtest_lookback_days', 30))
        if not pm:
            LOG.info('交易数据不足，跳过AI优化')
            return current_params
        
        if self.market_regime_shift(market_analysis):
            LOG.warning('市场状态突变，暂停AI优化')
            return current_params
        
        LOG.info(f'开始AI参数优化: {reason}')
        new_params = self.safe_optimization(current_params, pm, market_analysis)
        
        if new_params and new_params != current_params:
            self.last_optimization_time = time.time()
            self.performance_metrics['parameter_history'].append({
                'timestamp': datetime.now().isoformat(),
                'old_params': current_params,
                'new_params': new_params,
                'performance': pm,
                'optimization_reason': reason
            })
            self.save_performance_metrics()
            LOG.info('AI参数优化完成')
        
        return new_params if new_params else current_params
    
    def get_ai_optimization(self, current_params, performance_metrics, market_analysis):
        prompt = self._build_prompt(current_params, performance_metrics, market_analysis)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是量化交易参数优化专家，只返回JSON格式结果，不要解释。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        
        try:
            response = SESSION.post(self.base_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            # 修复：防止API密钥在日志中泄露
            ai_response = response.json()['choices'][0]['message']['content']
            # 记录响应但不包含敏感信息
            LOG.debug("AI优化响应接收成功")
            
            new_params = self._parse_ai_response(ai_response)
            return new_params
        except Exception as e:
            # 修复：在错误日志中不包含API密钥
            error_msg = str(e).replace(self.api_key, '***') if self.api_key else str(e)
            LOG.error(f'AI优化失败: {error_msg}')
            return None
    
    def _build_prompt(self, current_params, performance_metrics, market_analysis):
        validation_threshold = self.config_manager.get_config_value('ai_optimization.validation_threshold', 0.8)
        required_improvement = (1/validation_threshold - 1) * 100
        
        return f"""
请基于以下交易系统表现和市场分析，优化交易参数。请返回JSON格式的优化后参数。

重要验证标准（您的建议将基于以下标准验证）：
1. 回测性能必须比当前参数提升至少{required_improvement:.1f}%
2. 参数逻辑必须合理（止损<止盈，RSI超卖<超买）
3. 必须适应当前市场状态：{market_analysis['current_regime']}
4. 风险指标必须在可接受范围内

当前参数：
{json.dumps(current_params, indent=2)}

近期表现（{performance_metrics['lookback_days']}天）：
- 总交易次数: {performance_metrics['total_trades']}
- 总盈亏: {performance_metrics['total_pnl']:.2f} USDT
- 胜率: {performance_metrics['win_rate']:.1%}
- 平均盈利: {performance_metrics['avg_win']:.2f} USDT
- 平均亏损: {performance_metrics['avg_loss']:.2f} USDT
- 盈亏比: {performance_metrics['profit_factor']:.2f}
- 最大回撤: {performance_metrics.get('max_drawdown', 0):.2%}
- 趋势策略交易: {performance_metrics['trending_trades']}次, 盈亏: {performance_metrics['trending_pnl']:.2f} USDT
- 网格策略交易: {performance_metrics['ranging_trades']}次, 盈亏: {performance_metrics['ranging_pnl']:.2f} USDT

当前市场分析：
- 状态: {market_analysis['current_regime']}
- 趋势强度: {market_analysis['trend_strength']:.2f}
- 波动性状态: {market_analysis['volatility_regime']}
- RSI: {market_analysis['rsi']:.1f}
- ADX: {market_analysis['adx']:.1f}

请根据以上信息优化参数，确保：
1. 单次调整幅度不超过30%
2. 参数组合逻辑合理
3. 针对当前市场弱点进行改进
4. 平衡风险与收益

返回的JSON格式：
{{
  "base_trend_unit": 数值, // 趋势基础仓位大小 (范围: 50-300)
  "base_grid_unit": 数值, // 网格基础仓位大小 (范围: 20-150)
  "trend_stop_atr": 数值, // 趋势止损ATR倍数 (范围: 1.0-4.0)
  "grid_stop_atr": 数值, // 网格止损ATR倍数 (范围: 0.5-3.0)
  "grid_take_profit_atr": 数值, // 网格止盈ATR倍数 (范围: 1.0-4.0)
  "rsi_oversold": 数值, // RSI超卖阈值 (范围: 20-40)
  "rsi_overbought": 数值, // RSI超买阈值 (范围: 60-80)
  "adx_trend_threshold": 数值, // ADX趋势阈值 (范围: 15-35)
  "max_position_ratio": 数值 // 最大仓位比例 (范围: 0.5-3.0)
}}

请确保所有数值都在合理范围内，并显著改善系统表现。
"""
    
    def _parse_ai_response(self, text):
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                params = json.loads(json_str)
                return self.validate_params(params)
            else:
                LOG.error("无法从AI响应中提取JSON")
                return None
        except Exception as e:
            LOG.error(f"解析AI响应失败: {e}")
            return None
    
    def validate_params(self, params):
        validated = {}
        for key, (min_val, max_val) in self.safety_limits.items():
            if key in params:
                validated[key] = np.clip(params[key], min_val, max_val)
        return validated

# ========= 自动残仓处理 =========
class AutoStrayPositionHandler:
    """自动处理残仓"""
    
    def __init__(self, trading_bot):
        self.bot = trading_bot
    
    def auto_handle_stray_positions(self):
        LOG.info('开始自动残仓检测...')
        handled_symbols = set()
        
        for symbol in CFG['symbols']:
            try:
                if symbol in self.bot.suspended_symbols:
                    continue
                
                long_p, short_p = self.bot.ex.fetch_positions(symbol)
                actual_long = abs(float((long_p or {}).get('positionAmt', 0)))
                actual_short = abs(float((short_p or {}).get('positionAmt', 0)))
                
                recorded_long = self.bot.pos.get(symbol, {}).get('LONG', {}).get('units', 0)
                recorded_short = self.bot.pos.get(symbol, {}).get('SHORT', {}).get('units', 0)
                
                if (actual_long > 0 and recorded_long == 0) or (actual_short > 0 and recorded_short == 0):
                    LOG.warning(f'检测到{symbol}残仓，开始自动处理...')
                    success = self._handle_stray_position(symbol, actual_long, actual_short)
                    if success:
                        handled_symbols.add(symbol)
                        LOG.info(f'{symbol}残仓自动处理完成')
                    else:
                        LOG.error(f'{symbol}残仓自动处理失败，暂停该交易对')
                        self.bot.suspended_symbols.add(symbol)
                        
            except Exception as e:
                LOG.error(f'处理{symbol}残仓时出错: {e}')
                continue
        
        if handled_symbols:
            LOG.info(f'自动残仓处理完成: {list(handled_symbols)}')
        else:
            LOG.info('未发现需要处理的残仓')
    
    def _handle_stray_position(self, symbol, actual_long, actual_short):
        try:
            success = True
            if actual_long > 0:
                if not self._close_stray_position(symbol, 'LONG', actual_long):
                    success = success and self._incorporate_stray_position(symbol, 'LONG', actual_long)
            if actual_short > 0:
                if not self._close_stray_position(symbol, 'SHORT', actual_short):
                    success = success and self._incorporate_stray_position(symbol, 'SHORT', actual_short)
            return success
        except Exception as e:
            LOG.error(f'处理{symbol}残仓异常: {e}')
            return False
    
    def _close_stray_position(self, symbol, side, units):
        try:
            LOG.info(f'尝试平掉{symbol}{side}残仓，数量: {units}')
            success = self.bot.ex.close_all_positions(symbol, side, side, units, 0)
            if success:
                LOG.info(f'{symbol}{side}残仓平仓成功')
                return True
            else:
                LOG.warning(f'{symbol}{side}残仓平仓失败')
                return False
        except Exception as e:
            LOG.error(f'平仓{symbol}{side}残仓失败: {e}')
            return False
    
    def _incorporate_stray_position(self, symbol, side, units):
        try:
            LOG.info(f'尝试将{symbol}{side}残仓纳入系统管理')
            price = float(self.bot.ex.fetch_ticker(symbol)['last'])
            
            if symbol not in self.bot.pos:
                self.bot.pos[symbol] = {
                    'LONG': {'units': 0, 'entry_price': 0, 'stop_loss': 0, 'regime': '', 'unit_usdt': 0},
                    'SHORT': {'units': 0, 'entry_price': 0, 'stop_loss': 0, 'regime': '', 'unit_usdt': 0}
                }
            
            if side == 'LONG':
                stop_loss = price * 0.95
            else:
                stop_loss = price * 1.05
            
            self.bot.pos[symbol][side] = {
                'units': units,
                'entry_price': price,
                'stop_loss': stop_loss,
                'regime': 'auto_recovered',
                'unit_usdt': units * price
            }
            save_pos(self.bot.pos)
            LOG.info(f'{symbol}{side}残仓成功纳入系统管理')
            return True
        except Exception as e:
            LOG.error(f'纳入{symbol}{side}残仓失败: {e}')
            return False

# ========= 交易机器人 (Gate.io版本) =========
class AITradingBot:
    def __init__(self):
        # 先初始化激进度控制系统
        self.aggression_controller = AggressionController()
        self.config_manager = ThreadSafeConfigManager(self.aggression_controller)
        self.command_receiver = CommandReceiver(self.aggression_controller, self.config_manager)
        
        # 初始化权重系统
        self.weight_system = WeightedTradingSystem(self.config_manager)
        
        # 初始化配置覆盖
        self.config_manager.update_config_overrides()
        
        # 初始化Gate.io交易所
        self.ex = Exchange(self.config_manager)
        self.pos = load_pos()
        self.last_regime = {}
        self.max_equity = 0
        today = datetime.now().date().isoformat()
        self.daily_equity = {today: self.ex.equity()}
        
        # 启动命令监听
        self.command_receiver.start_listening()
        
        self.ai_optimizer = SafeAIOptimizer(self.config_manager)
        self.ai_position_manager = AIPositionManager(self.ex, self.ai_optimizer, self.config_manager)
        
        # 使用金融市场验证过的参数
        self.dynamic_params = {
            'base_trend_unit': 150,      # 增加趋势仓位
            'base_grid_unit': 80,        # 增加网格仓位
            'trend_stop_atr': 1.8,       # 更紧的止损
            'grid_stop_atr': 1.2,        # 更紧的网格止损
            'grid_take_profit_atr': 2.5, # 合理的止盈
            'rsi_oversold': 28,          # 更严格的超卖
            'rsi_overbought': 72,        # 更严格的超买
            'adx_trend_threshold': 22,   # 降低趋势阈值
            'max_position_ratio': 2.0    # 适中的杠杆
        }
        
        self.consecutive_errors = 0
        self.suspended_symbols = set()
        self.auto_monitor = AutoSystemMonitor(self)
        self.stray_handler = AutoStrayPositionHandler(self)
        
        # 初始化AI权重优化器
        self.ai_weight_optimizer = AIWeightOptimizer(self.weight_system, self.ai_optimizer)
        
        self.auto_initialize()
        LOG.info('✅ Gate.io AI增强交易机器人初始化完成')
        self.command_receiver._show_detailed_status()
    
    def auto_initialize(self):
        LOG.info('开始自动初始化流程...')
        self.stray_handler.auto_handle_stray_positions()
        self.auto_monitor.start_monitoring()
        self.sync_positions_on_startup()
        LOG.info('自动初始化流程完成')
    
    def calculate_signal_strength(self, symbol, mtf, strategy_type):
        """计算信号强度权重"""
        main_df = mtf[CFG['main_timeframe']]
        current_row = main_df.iloc[-1]
        
        weights = self.weight_system.condition_weights.copy()
        current_aggression = self.aggression_controller.current_level
        aggression_config = self.aggression_controller.aggression_multipliers[current_aggression]
        
        # 应用激进度乘数到权重
        for key in weights:
            weights[key] *= aggression_config['weight_multiplier']
        
        signal_scores = {
            'LONG': 0.0,
            'SHORT': 0.0
        }
        
        # 1. 趋势强度权重计算
        trend_score = self._calculate_trend_strength_weight(main_df)
        signal_scores['LONG'] += trend_score * weights['trend_strength']
        signal_scores['SHORT'] += trend_score * weights['trend_strength']
        
        # 2. 动量权重计算
        momentum_scores = self._calculate_momentum_weight(main_df)
        signal_scores['LONG'] += momentum_scores['LONG'] * weights['momentum']
        signal_scores['SHORT'] += momentum_scores['SHORT'] * weights['momentum']
        
        # 3. 波动率权重计算
        vol_score = self._calculate_volatility_weight(main_df)
        signal_scores['LONG'] += vol_score * weights['volatility']
        signal_scores['SHORT'] += vol_score * weights['volatility']
        
        # 4. RSI信号权重计算
        rsi_scores = self._calculate_rsi_weight(main_df)
        signal_scores['LONG'] += rsi_scores['LONG'] * weights['rsi_signal']
        signal_scores['SHORT'] += rsi_scores['SHORT'] * weights['rsi_signal']
        
        # 5. 布林带信号权重计算
        bb_scores = self._calculate_bb_weight(main_df)
        signal_scores['LONG'] += bb_scores['LONG'] * weights['bb_signal']
        signal_scores['SHORT'] += bb_scores['SHORT'] * weights['bb_signal']
        
        # 6. 成交量确认权重计算
        volume_scores = self._calculate_volume_weight(main_df)
        signal_scores['LONG'] += volume_scores['LONG'] * weights['volume_confirmation']
        signal_scores['SHORT'] += volume_scores['SHORT'] * weights['volume_confirmation']
        
        # 7. 市场状态权重计算
        regime_scores = self._calculate_regime_weight(main_df, strategy_type)
        signal_scores['LONG'] += regime_scores['LONG'] * weights['market_regime']
        signal_scores['SHORT'] += regime_scores['SHORT'] * weights['market_regime']
        
        # 调整阈值基于激进度
        adjusted_thresholds = {
            'open_long': self.weight_system.thresholds['open_long'] + aggression_config['threshold_adjust'],
            'open_short': self.weight_system.thresholds['open_short'] + aggression_config['threshold_adjust']
        }
        
        return signal_scores, adjusted_thresholds
    
    def _calculate_trend_strength_weight(self, df):
        """计算趋势强度权重"""
        current_row = df.iloc[-1]
        adx = current_row['adx']
        plus_di = current_row['plus_di']
        minus_di = current_row['minus_di']
        
        # ADX强度 (0-1)
        adx_strength = min(adx / 60, 1.0)  # ADX超过60视为最强
        
        # 方向强度 (0-1)
        di_strength = abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        
        # 综合趋势强度
        trend_strength = (adx_strength * 0.6 + di_strength * 0.4)
        
        return max(0, min(1, trend_strength))
    
    def _calculate_momentum_weight(self, df):
        """计算动量权重"""
        current_price = df['c'].iloc[-1]
        sma_20 = df['c'].rolling(20).mean().iloc[-1]
        sma_50 = df['c'].rolling(50).mean().iloc[-1]
        
        # 短期动量
        short_momentum = (current_price - df['c'].iloc[-5]) / df['c'].iloc[-5]
        
        # 均线排列
        ma_alignment = 1.0 if sma_20 > sma_50 else -1.0
        
        scores = {
            'LONG': max(0, (short_momentum + ma_alignment) / 2),
            'SHORT': max(0, (-short_momentum - ma_alignment) / 2)
        }
        
        return scores
    
    def _calculate_volatility_weight(self, df):
        """计算波动率权重 - 适中的波动率最好"""
        current_atr = df['atr'].iloc[-1]
        avg_atr = df['atr'].rolling(50).mean().iloc[-1]
        
        vol_ratio = current_atr / avg_atr
        
        # 波动率在0.8-1.5之间为最佳
        if 0.8 <= vol_ratio <= 1.5:
            return 1.0
        elif 0.5 <= vol_ratio < 0.8 or 1.5 < vol_ratio <= 2.0:
            return 0.5
        else:
            return 0.2
    
    def _calculate_rsi_weight(self, df):
        """计算RSI权重"""
        rsi = df['rsi'].iloc[-1]
        
        scores = {'LONG': 0.0, 'SHORT': 0.0}
        
        # RSI超卖区域
        if rsi < 30:
            scores['LONG'] = 1.0 - (rsi / 30)  # 越低分数越高
        elif rsi < 45:
            scores['LONG'] = 0.5 * (1 - (rsi - 30) / 15)
        
        # RSI超买区域  
        if rsi > 70:
            scores['SHORT'] = 1.0 - ((100 - rsi) / 30)  # 越高分数越高
        elif rsi > 55:
            scores['SHORT'] = 0.5 * (1 - (70 - rsi) / 15)
        
        return scores
    
    def _calculate_bb_weight(self, df):
        """修复布林带权重计算"""
        try:
            current_price = df['c'].iloc[-1]
            bb_upper = df['bb_up'].iloc[-1]
            bb_lower = df['bb_low'].iloc[-1]
            bb_mid = df['bb_mid'].iloc[-1]
            
            # 防止除零
            if (bb_upper - bb_lower) <= 0:
                return {'LONG': 0.0, 'SHORT': 0.0}
            
            # 计算价格在布林带中的位置 (0-1)
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            bb_position = max(0.0, min(1.0, bb_position))  # 钳制在0-1之间
            
            scores = {'LONG': 0.0, 'SHORT': 0.0}
            
            # 下轨附近做多
            if bb_position < 0.2:
                scores['LONG'] = 1.0 - (bb_position / 0.2)
            elif bb_position < 0.3:
                scores['LONG'] = 0.5 * (1 - (bb_position - 0.2) / 0.1)
            
            # 上轨附近做空
            if bb_position > 0.8:
                scores['SHORT'] = 1.0 - ((1 - bb_position) / 0.2)
            elif bb_position > 0.7:
                scores['SHORT'] = 0.5 * (1 - (0.8 - bb_position) / 0.1)
            
            return scores
        except Exception as e:
            LOG.error(f"布林带权重计算失败: {e}")
            return {'LONG': 0.0, 'SHORT': 0.0}
    
    def _calculate_volume_weight(self, df):
        """计算成交量确认权重"""
        current_volume = df['v'].iloc[-1]
        avg_volume = df['v'].rolling(20).mean().iloc[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # 成交量放大有利于趋势延续
        volume_strength = min(volume_ratio / 2.0, 1.0)  # 2倍以上成交量给满分
        
        return {'LONG': volume_strength, 'SHORT': volume_strength}
    
    def _calculate_regime_weight(self, df, strategy_type):
        """计算市场状态权重"""
        current_regime = df['regime'].iloc[-1]
        
        scores = {'LONG': 0.0, 'SHORT': 0.0}
        
        if strategy_type == 'trending':
            # 趋势策略在趋势市场中权重更高
            if current_regime == 'trending':
                scores['LONG'] = 1.0
                scores['SHORT'] = 1.0
            else:
                scores['LONG'] = 0.3
                scores['SHORT'] = 0.3
        else:
            # 网格策略在震荡市场中权重更高
            if current_regime == 'ranging':
                scores['LONG'] = 1.0
                scores['SHORT'] = 1.0
            else:
                scores['LONG'] = 0.3
                scores['SHORT'] = 0.3
        
        return scores
    
    def _execute_trade(self, symbol, side, signal_score, mtf, equity, regime):
        """执行交易 - 集成动态止损止盈"""
        try:
            main_df = mtf[CFG['main_timeframe']]
            price = float(main_df['c'].iloc[-1])
            atr = float(main_df['atr'].iloc[-1])
            
            # 获取市场分析用于动态止损
            market_analysis = self.analyze_market_for_ai(symbol, mtf)
            
            # 使用动态止损止盈计算
            stop_loss, take_profit = self.calculate_dynamic_stop_tp(
                symbol, side, price, atr, regime, 
                market_analysis.get('volatility_regime', 'normal')
            )
            
            # 基于信号强度调整仓位大小
            position_multiplier = 0.5 + (signal_score * 0.5)
            base_size = self.get_param('base_trend_unit') if regime == 'trending' else self.get_param('base_grid_unit')
            
            # 计算最终仓位大小
            position_size = self.calculate_position_size(
                symbol, 
                'trend_follow' if regime == 'trending' else 'grid_mean_reversion', 
                base_size * position_multiplier, 
                mtf
            )
            
            if position_size <= 0:
                LOG.warning(f"{symbol} 仓位计算为0，跳过交易")
                return
            
            # 检查对冲冲突
            if self.check_hedge_conflict(symbol, side):
                return
            
            # 下订单
            order_side = 'buy' if side == 'LONG' else 'sell'
            order = self.ex.place_market_order(symbol, order_side, position_size, side)
            
            if order:
                # 更新仓位记录
                if symbol not in self.pos:
                    self.pos[symbol] = {'LONG': {}, 'SHORT': {}}
                
                self.pos[symbol][side] = {
                    'units': position_size / price,
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'regime': regime,
                    'unit_usdt': position_size,
                    'signal_score': signal_score,
                    'timestamp': datetime.now().isoformat()
                }
                save_pos(self.pos)
                
                # 记录详细交易
                self.record_detailed_trade(
                    symbol, side, price, price, 
                    position_size / price, 0, regime, signal_score,
                    self.weight_system.condition_weights,
                    market_analysis,
                    {'action': 'open', 'position_size': position_size}
                )
                
                LOG.info(f"✅ {symbol} {side} 开仓 - 信号强度: {signal_score:.2f}, "
                        f"仓位: {position_size:.2f} USDT, 止损: {stop_loss:.2f}, 止盈: {take_profit:.2f}")
        except Exception as e:
            LOG.error(f"执行交易失败: {e}")
    
    def funding_filter(self, symbol, side):
        """资金费率过滤器"""
        try:
            if self.funding_imminent(symbol):
                LOG.warning(f'{symbol} 资金费结算前5分钟，不开仓')
                return False
            
            if self.ex.funding_manager.should_avoid_trading(symbol, side,
                self.get_param('realistic_trading.funding_avoid_hours')):
                return False
            
            return True
        except Exception as e:
            LOG.error(f"资金费率检查失败: {e}")
            return False
    
    def _log_signal_details(self, symbol, strategy, long_score, short_score, thresholds):
        """记录信号详情"""
        try:
            if long_score >= thresholds['open_long'] or short_score >= thresholds['open_short']:
                LOG.info(f"🎯 {symbol} {strategy}信号 - 多头: {long_score:.2f}, 空头: {short_score:.2f}, "
                        f"阈值: {thresholds['open_long']:.2f}")
                
                # 详细记录各条件得分
                if hasattr(self, 'last_signal_analysis'):
                    analysis = self.last_signal_analysis.get(symbol, {})
                    for condition, score in analysis.items():
                        LOG.debug(f"    {condition}: {score:.2f}")
        except Exception as e:
            LOG.debug(f"记录信号详情失败: {e}")
    
    def trend_logic(self, symbol, mtf, equity):
        """基于权重的趋势逻辑"""
        if symbol in self.suspended_symbols:
            return
        
        # 计算信号强度
        signal_scores, thresholds = self.calculate_signal_strength(symbol, mtf, 'trending')
        
        long_score = signal_scores['LONG']
        short_score = signal_scores['SHORT']
        
        # 记录信号详情用于调试
        self._log_signal_details(symbol, 'TREND', long_score, short_score, thresholds)
        
        # 检查资金费率
        if long_score >= thresholds['open_long']:
            if not self.funding_filter(symbol, 'LONG'):
                return
            self._execute_trade(symbol, 'LONG', long_score, mtf, equity, 'trending')
        
        elif short_score >= thresholds['open_short']:
            if not self.funding_filter(symbol, 'SHORT'):
                return
            self._execute_trade(symbol, 'SHORT', short_score, mtf, equity, 'trending')
    
    def range_logic(self, symbol, mtf, equity):
        """基于权重的网格逻辑"""
        if symbol in self.suspended_symbols:
            return
        
        # 计算信号强度
        signal_scores, thresholds = self.calculate_signal_strength(symbol, mtf, 'ranging')
        
        long_score = signal_scores['LONG']
        short_score = signal_scores['SHORT']
        
        # 记录信号详情用于调试
        self._log_signal_details(symbol, 'RANGE', long_score, short_score, thresholds)
        
        # 网格策略使用相同的阈值但不同的资金管理
        if long_score >= thresholds['open_long']:
            if not self.funding_filter(symbol, 'LONG'):
                return
            self._execute_trade(symbol, 'LONG', long_score, mtf, equity, 'ranging')
        
        elif short_score >= thresholds['open_short']:
            if not self.funding_filter(symbol, 'SHORT'):
                return
            self._execute_trade(symbol, 'SHORT', short_score, mtf, equity, 'ranging')
    
    def check_stop_loss_take_profit(self, symbol, mtf):
        """检查止损止盈 - 带详细记录"""
        if symbol not in self.pos:
            return
        
        main_df = mtf[CFG['main_timeframe']]
        current_price = float(main_df['c'].iloc[-1])
        pos_data = self.pos[symbol]
        
        for side in ['LONG', 'SHORT']:
            if side in pos_data and pos_data[side].get('units', 0) > 0:
                entry_price = pos_data[side]['entry_price']
                stop_loss = pos_data[side]['stop_loss']
                take_profit = pos_data[side]['take_profit']
                units = pos_data[side]['units']
                unit_usdt = pos_data[side]['unit_usdt']
                signal_score = pos_data[side].get('signal_score', 0)
                regime = pos_data[side].get('regime', '')
                
                should_close = False
                close_reason = ""
                
                if side == 'LONG':
                    if current_price <= stop_loss:
                        should_close = True
                        close_reason = f"止损 ({current_price} <= {stop_loss})"
                    elif current_price >= take_profit:
                        should_close = True
                        close_reason = f"止盈 ({current_price} >= {take_profit})"
                else:  # SHORT
                    if current_price >= stop_loss:
                        should_close = True
                        close_reason = f"止损 ({current_price} >= {stop_loss})"
                    elif current_price <= take_profit:
                        should_close = True
                        close_reason = f"止盈 ({current_price} <= {take_profit})"
                
                if should_close:
                    success = self.ex.close_all_positions(symbol, side, side, units, unit_usdt)
                    if success:
                        # 记录详细交易结果
                        pnl = self.calculate_pnl(symbol, side, entry_price, 
                                               current_price, units, unit_usdt)
                        
                        # 获取市场分析
                        market_analysis = self.analyze_market_for_ai(symbol, mtf)
                        
                        # 记录详细交易
                        self.record_detailed_trade(
                            symbol, side, entry_price, current_price, 
                            units, pnl, regime, signal_score,
                            self.weight_system.condition_weights,
                            market_analysis,
                            {'close_reason': close_reason, 'close_type': 'stop_tp'}
                        )
                        
                        LOG.info(f"✅ {symbol} {side} 平仓 - {close_reason}, PnL: {pnl:.2f}")
                        
                        # 清除仓位记录
                        pos_data[side] = {'units': 0, 'entry_price': 0, 'stop_loss': 0, 
                                        'take_profit': 0, 'regime': '', 'unit_usdt': 0}
                        save_pos(self.pos)
    
    def calculate_pnl(self, symbol, side, entry_price, exit_price, units, position_size):
        """计算盈亏 - 统一版本"""
        try:
            market = self.ex.ex.market(symbol)
            contract_size = float(market['contractSize'])
            if side == 'LONG':
                pnl = (exit_price - entry_price) * units * contract_size
            else:
                pnl = (entry_price - exit_price) * units * contract_size
            
            # 记录详细计算
            LOG.debug(f"{symbol} {side} PnL计算: 入场{entry_price:.2f}, 出场{exit_price:.2f}, "
                     f"数量{units:.4f}, 合约大小{contract_size}, 结果{pnl:.2f}")
            return pnl
        except Exception as e:
            LOG.error(f'盈亏计算失败: {e}')
            # 备用计算方法
            position_value = units * position_size
            if side == 'LONG':
                return (exit_price - entry_price) / entry_price * position_value
            else:
                return (entry_price - exit_price) / entry_price * position_value
    
    def record_detailed_trade(self, symbol, side, entry_price, exit_price, units, pnl, regime, 
                             signal_score, weights, market_analysis, decision_details):
        """详细的交易记录"""
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': units,
            'pnl': pnl,
            'regime': regime,
            'parameters': self.dynamic_params.copy(),
            # 新增详细记录
            'signal_strength': signal_score,
            'weight_config': weights.copy(),
            'market_analysis': {
                'regime': market_analysis.get('current_regime'),
                'volatility': market_analysis.get('volatility_regime'),
                'trend_strength': market_analysis.get('trend_strength'),
                'rsi': market_analysis.get('rsi'),
                'adx': market_analysis.get('adx')
            },
            'decision_details': decision_details,
            'aggression_level': self.aggression_controller.current_level
        }
        
        self.ai_optimizer.memory_manager.record_trade(
            self.ai_optimizer.performance_metrics, 
            trade_record
        )
        self.ai_optimizer.save_performance_metrics()
    
    def run(self):
        LOG.info('🚀 Gate.io AI增强交易机器人启动')
        last_optimize = time.time()
        last_weight_optimize = time.time()
        last_aggression_check = time.time()
        
        while True:
            try:
                current_time = time.time()
                
                # 检查激进度变化
                if current_time - last_aggression_check > 30:
                    if self.aggression_controller.has_level_changed():
                        self.config_manager.update_config_overrides()
                        LOG.info("激进度配置已更新")
                        self._show_detailed_aggression_changes()
                        self._reassess_positions_for_aggression_change()
                    last_aggression_check = current_time
                
                # 权重优化 (每6小时)
                if current_time - last_weight_optimize > 6 * 3600:
                    if len(self.ai_optimizer.performance_metrics['trades']) >= 20:
                        performance_data = list(self.ai_optimizer.performance_metrics['trades'])[-50:]  # 最近50笔交易
                        if self.optimize_weights_with_validation(performance_data):
                            LOG.info("✅ AI权重优化完成")
                            self._log_current_weights()
                    last_weight_optimize = current_time
                
                # 放宽错误计数限制
                if self.consecutive_errors > 20:  # 从10提高到20
                    LOG.critical('连续错误过多，紧急退出')
                    raise SystemExit
                
                if not self.check_circuit():
                    LOG.error('熔断激活，暂停交易')
                    time.sleep(60)
                    continue
                
                hot_reload_cfg()
                equity = self.ex.equity()
                
                today = datetime.now().date().isoformat()
                if today not in self.daily_equity:
                    self.daily_equity = {today: equity}
                
                if equity > self.max_equity:
                    self.max_equity = equity
                
                # 处理每个交易对，分别捕获异常
                for symbol in CFG['symbols']:
                    try:
                        self._process_symbol(symbol, current_time, last_optimize)
                        time.sleep(0.5)  # 避免API限流
                        
                    except ccxt.NetworkError as e:
                        LOG.warning(f'{symbol} 网络错误: {e}')
                        self.consecutive_errors += 1
                        time.sleep(5)
                        
                    except ccxt.ExchangeError as e:
                        error_str = str(e).lower()
                        if 'rate limit' in error_str:
                            LOG.warning(f'{symbol} 频率限制，等待后继续')
                            time.sleep(10)
                        else:
                            LOG.error(f'{symbol} 交易所错误: {e}')
                            self.consecutive_errors += 1
                            time.sleep(5)
                            
                    except Exception as e:
                        error_type = EnhancedExceptionHandler.handle_trading_exception(e, symbol, '主循环')
                        if error_type == 'CRITICAL':
                            self.suspended_symbols.add(symbol)
                            LOG.error(f'暂停{symbol}交易')
                        
                        self.consecutive_errors += 1
                        time.sleep(5)
                        continue
                
                self.consecutive_errors = 0
                time.sleep(CFG['loop_sec'])
                
            except KeyboardInterrupt:
                LOG.info('用户中断')
                break
            except Exception as e:
                LOG.critical(f'主循环异常: {e}')
                time.sleep(30)
    
    def optimize_weights_with_validation(self, performance_data):
        """带验证的权重优化"""
        if self.ai_weight_optimizer.optimize_weights(performance_data):
            # 验证新权重的合理性
            new_weights = self.weight_system.condition_weights
            total_weight = sum(new_weights.values())
            
            if abs(total_weight - 1.0) > 0.01:  # 权重总和应该接近1
                LOG.error(f"权重归一化异常: {total_weight}, 重置为默认权重")
                self.weight_system.condition_weights = self.weight_system._initialize_weights()
                return False
            
            # 保存优化历史
            if hasattr(self.ai_weight_optimizer, 'optimization_history') and self.ai_weight_optimizer.optimization_history:
                optimization_record = {
                    'timestamp': datetime.now().isoformat(),
                    'old_weights': self.ai_weight_optimizer.optimization_history[-1]['old_weights'],
                    'new_weights': new_weights,
                    'performance_data': performance_data[-10:]
                }
                
                # 保存到性能指标
                self.ai_optimizer.performance_metrics.setdefault('weight_optimizations', []).append(optimization_record)
                self.ai_optimizer.save_performance_metrics()
            
            LOG.info("✅ AI权重优化完成并验证")
            return True
        return False
    
    def _process_symbol(self, symbol, current_time, last_optimize):
        """处理单个交易对 - 细化异常处理"""
        if symbol in self.suspended_symbols:
            return
        
        mtf = self.ex.fetch_mtf(symbol)
        if not mtf:
            LOG.warning(f'{symbol} 无K线数据')
            return
        
        main_df = mtf[CFG['main_timeframe']]
        if len(main_df) < 100:
            LOG.warning(f'{symbol} 数据不足')
            return
        
        # 检查止损止盈
        self.check_stop_loss_take_profit(symbol, mtf)
        
        current_row = main_df.iloc[-1]
        new_regime = current_row['regime']
        
        # 检查AI优化
        market_analysis = self.analyze_market_for_ai(symbol, mtf)
        if current_time - last_optimize > 3600:
            new_params = self.ai_optimizer.optimize_parameters(self.dynamic_params, market_analysis)
            if new_params:
                self.dynamic_params = new_params
                last_optimize = current_time
        
        # 状态切换平仓
        if new_regime != self.last_regime.get(symbol):
            self.regime_switch_flat(symbol, new_regime)
            self.last_regime[symbol] = new_regime
        
        # 执行策略
        if new_regime == 'trending':
            self.trend_logic(symbol, mtf, self.ex.equity())
        else:
            self.range_logic(symbol, mtf, self.ex.equity())
    
    def sync_positions_on_startup(self):
        """增强启动仓位同步检查"""
        LOG.info('启动仓位同步检查...')
        with position_lock:
            for symbol in CFG['symbols']:
                try:
                    long_p, short_p = self.ex.fetch_positions(symbol)
                    actual_long = abs(float((long_p or {}).get('positionAmt', 0)))
                    actual_short = abs(float((short_p or {}).get('positionAmt', 0)))
                    
                    recorded_long = self.pos.get(symbol, {}).get('LONG', {}).get('units', 0)
                    recorded_short = self.pos.get(symbol, {}).get('SHORT', {}).get('units', 0)
                    
                    # 处理所有不一致情况
                    inconsistencies = []
                    
                    # 情况1: 记录有仓位但实际没有
                    if recorded_long > 0 and actual_long == 0:
                        inconsistencies.append(f"LONG记录{recorded_long}但实际0")
                        self.pos[symbol]['LONG'] = {'units': 0, 'entry_price': 0, 'stop_loss': 0, 'regime': '', 'unit_usdt': 0}
                    
                    if recorded_short > 0 and actual_short == 0:
                        inconsistencies.append(f"SHORT记录{recorded_short}但实际0")
                        self.pos[symbol]['SHORT'] = {'units': 0, 'entry_price': 0, 'stop_loss': 0, 'regime': '', 'unit_usdt': 0}
                    
                    # 情况2: 实际有仓位但记录没有 (残仓)
                    if actual_long > 0 and recorded_long == 0:
                        inconsistencies.append(f"LONG实际{actual_long}但记录0")
                        if not self._handle_stray_position_on_startup(symbol, 'LONG', actual_long):
                            LOG.error(f"无法处理{symbol} LONG残仓")
                    
                    if actual_short > 0 and recorded_short == 0:
                        inconsistencies.append(f"SHORT实际{actual_short}但记录0")
                        if not self._handle_stray_position_on_startup(symbol, 'SHORT', actual_short):
                            LOG.error(f"无法处理{symbol} SHORT残仓")
                    
                    if inconsistencies:
                        LOG.warning(f'{symbol} 仓位不一致: {", ".join(inconsistencies)}')
                        
                except Exception as e:
                    LOG.error(f'同步{symbol}仓位时出错: {e}')
                    continue
            
            save_pos(self.pos)
            LOG.info('仓位同步完成')
    
    def _handle_stray_position_on_startup(self, symbol, side, units):
        """启动时的残仓处理"""
        try:
            LOG.warning(f'检测到{symbol} {side}残仓，数量: {units}')
            
            # 尝试平仓
            success = self.ex.close_all_positions(symbol, side, side, units, 0)
            if success:
                LOG.info(f'{symbol} {side}残仓平仓成功')
                return True
            else:
                # 平仓失败，尝试纳入管理
                return self._incorporate_stray_position(symbol, side, units)
        except Exception as e:
            LOG.error(f'处理{symbol} {side}残仓失败: {e}')
            return False
    
    def _incorporate_stray_position(self, symbol, side, units):
        """将残仓纳入系统管理"""
        try:
            LOG.info(f'尝试将{symbol}{side}残仓纳入系统管理')
            price = float(self.ex.fetch_ticker(symbol)['last'])
            
            if symbol not in self.pos:
                self.pos[symbol] = {
                    'LONG': {'units': 0, 'entry_price': 0, 'stop_loss': 0, 'regime': '', 'unit_usdt': 0},
                    'SHORT': {'units': 0, 'entry_price': 0, 'stop_loss': 0, 'regime': '', 'unit_usdt': 0}
                }
            
            if side == 'LONG':
                stop_loss = price * 0.95
            else:
                stop_loss = price * 1.05
            
            self.pos[symbol][side] = {
                'units': units,
                'entry_price': price,
                'stop_loss': stop_loss,
                'regime': 'auto_recovered',
                'unit_usdt': units * price
            }
            save_pos(self.pos)
            LOG.info(f'{symbol}{side}残仓成功纳入系统管理')
            return True
        except Exception as e:
            LOG.error(f'纳入{symbol}{side}残仓失败: {e}')
            return False
    
    def get_param(self, name):
        try:
            return self.config_manager.get_config_value(name)
        except:
            return self.dynamic_params.get(name, CFG.get(name))
    
    def calculate_position_size(self, symbol, strategy_type, base_size, mtf):
        if not self.config_manager.get_config_value('ai_position_management.enabled', True):
            equity = self.ex.equity()
            tier = self.ai_position_manager._get_capital_tier(equity)
            base_risk_ratios = {'micro': 0.04, 'small': 0.03, 'medium': 0.025, 'large': 0.02, 'institutional': 0.015}
            base_risk = base_risk_ratios[tier]
            return equity * base_risk
        
        market_analysis = self.analyze_market_for_ai(symbol, mtf)
        return self.ai_position_manager.get_ai_optimized_position_size(symbol, strategy_type, market_analysis)
    
    def funding_imminent(self, symbol):
        """精确的资金费率时间检查"""
        try:
            fr, next_t = self.ex.fetch_funding_rate(symbol)
            if not next_t or next_t <= 0:
                return False
            
            current_time = datetime.now().timestamp()
            seconds_left = (next_t / 1000) - current_time
            
            # 动态时间窗口：根据交易所和当前市场状况调整
            base_window = 300  # 5分钟基础窗口
            
            # 根据资金费率大小调整窗口
            if abs(fr) > 0.001:  # 资金费率较高时提前避免
                base_window = 600  # 10分钟
            
            # 根据市场波动调整
            try:
                mtf = self.ex.fetch_mtf(symbol)
                if mtf:
                    main_df = mtf[CFG['main_timeframe']]
                    atr_ratio = main_df['atr'].iloc[-1] / main_df['atr'].rolling(50).mean().iloc[-1]
                    if atr_ratio > 1.5:  # 高波动市场
                        base_window = 900  # 15分钟
            except:
                pass
            
            return 0 < seconds_left < base_window
            
        except Exception as e:
            LOG.error(f'资金费检查失败: {e}')
            return False
    
    def day_range_filter(self, df: pd.DataFrame):
        day_high = df['h'].iloc[-1]
        day_low = df['l'].iloc[-1]
        day_range = (day_high - day_low) / df['c'].iloc[-1]
        return day_range < 0.08
    
    def adapt_leverage(self, symbol):
        try:
            market = self.ex.ex.market(symbol)
            max_leverage = market['limits']['leverage']['max']
            if CFG['lev'] > max_leverage:
                LOG.warning(f'交易所下调杠杆，当前{CFG["lev"]} > 允许{max_leverage}，强制降杠杆')
                CFG['lev'] = max_leverage
                retry(self.ex.ex.set_leverage, max_leverage, symbol)
        except Exception as e:
            LOG.error(f'杠杆调整失败: {e}')
    
    def check_add_spacing(self, symbol, side, price, atr, pos):
        last_add_key = f'last_add_price_{side}'
        last_price = pos.get(last_add_key, 0)
        if last_price and abs(price - last_price) < 0.2 * atr:
            LOG.warning('价格未拉开，跳过加仓')
            return False
        pos[last_add_key] = price
        return True
    
    def check_hedge_conflict(self, symbol, side):
        pos = self.pos.get(symbol, {})
        if side == 'LONG' and pos.get('SHORT', {}).get('units', 0) > 0:
            LOG.warning('双向冲突，先平空再开多')
            return True
        if side == 'SHORT' and pos.get('LONG', {}).get('units', 0) > 0:
            LOG.warning('双向冲突，先平多再开空')
            return True
        return False
    
    def exchange_health(self):
        return exchange_health(self.ex.ex)
    
    def emergency_snapshot(self, symbol, mtf):
        pos_data = self.pos.get(symbol, {})
        safe_pos = {}
        for side in ['LONG', 'SHORT']:
            if side in pos_data:
                safe_pos[side] = {
                    'units': pos_data[side].get('units', 0),
                    'regime': pos_data[side].get('regime', '')
                }
        
        main_df = mtf[CFG['main_timeframe']]
        snap = {
            'ts': datetime.now().isoformat(),
            'symbol': symbol,
            'price': float(main_df['c'].iloc[-1]),
            'pos': safe_pos,
            'atr': float(main_df['atr'].iloc[-1]),
            'equity': self.ex.equity()
        }
        filename = f'emergency_gate_{symbol}_{int(time.time())}.json'
        with open(filename, 'w') as f:
            json.dump(snap, f, indent=2)
        LOG.info(f'紧急快照已保存: {filename}')
    
    def analyze_market_for_ai(self, symbol, mtf):
        main_df = mtf[CFG['main_timeframe']]
        row = main_df.iloc[-1]
        
        analysis = {
            'symbol': symbol,
            'current_regime': row['regime'],
            'price': float(row['c']),
            'atr': float(row['atr']),
            'atr_ratio_50ma': float(row['atr'] / main_df['atr'].rolling(50).mean().iloc[-1]),
            'rsi': float(row['rsi']),
            'adx': float(row['adx']),
            'bb_width': float(row['bb_width']),
            'trend_strength': self.trend_strength(mtf),
            'volatility_regime': 'high' if float(row['atr'] / main_df['atr'].rolling(50).mean().iloc[-1]) > 1.2 else 'low',
            'df': main_df
        }
        return analysis
    
    def trend_strength(self, mtf):
        strengths = []
        for tf, df in mtf.items():
            if len(df) < 20: continue
            row = df.iloc[-1]
            adx_s = min(row['adx'] / 50, 1.0) if not pd.isna(row['adx']) else 0
            di_s = abs(row['plus_di'] - row['minus_di']) / 100
            bb_pos = (row['c'] - row['bb_low']) / (row['bb_up'] - row['bb_low'])
            bb_s = 1 - 2 * abs(bb_pos - 0.5)
            strengths.append(adx_s * 0.4 + di_s * 0.3 + bb_s * 0.3)
        return np.mean(strengths) if strengths else 0
    
    def calculate_pnl_with_funding(self, symbol, side, entry_price, exit_price, units, position_size, entry_time, exit_time):
        try:
            if side == 'LONG':
                price_pnl = (exit_price - entry_price) * units
            else:
                price_pnl = (entry_price - exit_price) * units
            
            funding_cost = self.ex.funding_manager.calculate_funding_cost(symbol, position_size, entry_time, exit_time)
            if funding_cost > 0:
                self.ex.funding_manager.record_funding_payment(symbol, side, -funding_cost, exit_time)
            
            net_pnl = price_pnl - funding_cost
            LOG.info(f'{symbol} {side} 盈亏: 价格{price_pnl:.2f} - 资金费用{funding_cost:.2f} = 净盈亏{net_pnl:.2f}')
            return net_pnl
        except Exception as e:
            LOG.error(f'含资金费用盈亏计算失败: {e}')
            return self.calculate_pnl(symbol, side, entry_price, exit_price, units, position_size)
    
    def regime_switch_flat(self, symbol, new_reg):
        old = self.last_regime.get(symbol, '')
        if old and old != new_reg:
            LOG.info(f'{symbol} 状态切换 {old}→{new_reg} 先平旧仓')
            pos = self.pos.get(symbol, {})
            for pk in ('LONG', 'SHORT'):
                units = pos.get(pk, {}).get('units', 0)
                if units > 0:
                    unit_usdt = pos[pk].get('unit_usdt', 
                        self.get_param('base_trend_unit') if pos[pk]['regime'] == 'trending' else self.get_param('base_grid_unit'))
                    success = self.ex.close_all_positions(symbol, pk, pk, units, unit_usdt)
                    if success:
                        pos[pk] = {'units': 0, 'entry_price': 0, 'stop_loss': 0, 'regime': '', 'unit_usdt': 0}
            save_pos(self.pos)
    
    def check_circuit(self):
        return self.ex.circuit_breaker.check_circuit()
    
    def available_notional(self, symbol, equity):
        max_notional = equity * self.get_param('max_equity_risk')
        long_p, short_p = self.ex.fetch_positions(symbol)
        long_notional = float(long_p.get('notional', 0)) if long_p else 0
        short_notional = float(short_p.get('notional', 0)) if short_p else 0
        used = long_notional + short_notional
        
        open_orders = retry(self.ex.fetch_open_orders, symbol)
        frozen = sum(abs(float(o['amount'])) * float(o['price']) for o in open_orders)
        
        return max(max_notional - used - frozen, 0)
    
    def _show_detailed_aggression_changes(self):
        """显示激进度变化的详细信息"""
        level = self.aggression_controller.current_level
        detailed_config = self.aggression_controller.get_detailed_config(level)
        
        LOG.info(f"\n🎯 激进度已切换到级别 {level}")
        LOG.info(f"📈 主要变化:")
        LOG.info(f"  • 风险承受: {detailed_config['risk_parameters']['max_equity_risk']}")
        LOG.info(f"  • 仓位大小: {detailed_config['position_parameters']['base_trend_unit']} USDT")
        LOG.info(f"  • 开单阈值: {detailed_config['weight_parameters']['open_threshold']}")
        LOG.info(f"  • 信号权重: {detailed_config['weight_parameters']['weight_multiplier']}")
    
    def _log_current_weights(self):
        """记录当前权重配置"""
        weights = self.weight_system.condition_weights
        LOG.info("📊 当前条件权重配置:")
        for condition, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            LOG.info(f"  {condition}: {weight:.2f} ({weight*100:.1f}%)")
    
    def _reassess_positions_for_aggression_change(self):
        """激进度变化时重评估持仓"""
        LOG.info("🔄 激进度变化，重新评估当前持仓...")
        
        current_level = self.aggression_controller.current_level
        aggression_config = self.aggression_controller.get_current_config()
        changes_made = False
        
        for symbol in CFG['symbols']:
            if symbol not in self.pos:
                continue
                
            for side in ['LONG', 'SHORT']:
                position = self.pos[symbol].get(side, {})
                if position.get('units', 0) > 0:
                    position_size = position.get('unit_usdt', 0)
                    equity = self.ex.equity()
                    position_ratio = position_size / equity if equity > 0 else 0
                    
                    max_position_ratio = self.config_manager.get_config_value('max_position_ratio', 2.5)
                    
                    # 检查是否需要减仓
                    if position_ratio > max_position_ratio:
                        reduction_ratio = position_ratio / max_position_ratio
                        if reduction_ratio > 1.2:  # 超过20%才执行减仓
                            reduction_amount = position_size * (1 - 1/reduction_ratio)
                            LOG.warning(f"{symbol} {side} 仓位比例{position_ratio:.1%}超过限制，减仓{reduction_amount:.2f} USDT")
                            
                            # 执行减仓
                            if self._reduce_position(symbol, side, reduction_amount):
                                changes_made = True
                    
                    # 更新止损止盈
                    try:
                        mtf = self.ex.fetch_mtf(symbol)
                        if mtf:
                            main_df = mtf[CFG['main_timeframe']]
                            price = float(main_df['c'].iloc[-1])
                            atr = float(main_df['atr'].iloc[-1])
                            market_analysis = self.analyze_market_for_ai(symbol, mtf)
                            
                            new_stop, new_tp = self.calculate_dynamic_stop_tp(
                                symbol, side, price, atr, 
                                position.get('regime', 'trending'),
                                market_analysis.get('volatility_regime', 'normal')
                            )
                            
                            # 更新止损止盈
                            self.pos[symbol][side]['stop_loss'] = new_stop
                            self.pos[symbol][side]['take_profit'] = new_tp
                            changes_made = True
                            LOG.info(f"{symbol} {side} 止损更新为{new_stop:.2f}, 止盈{new_tp:.2f}")
                    except Exception as e:
                        LOG.error(f"更新{symbol} {side}止损失败: {e}")
        
        if changes_made:
            save_pos(self.pos)
            LOG.info("持仓重评估完成并保存")
        else:
            LOG.info("持仓无需调整")
    
    def _reduce_position(self, symbol, side, reduction_amount):
        """减少持仓"""
        try:
            current_units = self.pos[symbol][side].get('units', 0)
            if current_units <= 0:
                return False
            
            reduction_units = min(current_units, reduction_amount / self.pos[symbol][side].get('entry_price', 1))
            if reduction_units <= 0:
                return False
            
            # 执行减仓
            close_side = 'sell' if side == 'LONG' else 'buy'
            order = self.ex.place_market_order(symbol, close_side, reduction_amount, side)
            
            if order:
                # 更新仓位记录
                self.pos[symbol][side]['units'] -= reduction_units
                self.pos[symbol][side]['unit_usdt'] = self.pos[symbol][side]['units'] * self.pos[symbol][side].get('entry_price', 0)
                
                # 如果仓位为0，清除记录
                if self.pos[symbol][side]['units'] <= 0.001:
                    self.pos[symbol][side] = {'units': 0, 'entry_price': 0, 'stop_loss': 0, 'regime': '', 'unit_usdt': 0}
                
                LOG.info(f"✅ {symbol} {side} 减仓完成，减少{reduction_units:.4f}单位")
                return True
            
            return False
        except Exception as e:
            LOG.error(f"减仓失败: {e}")
            return False
    
    def calculate_dynamic_stop_tp(self, symbol, side, price, atr, regime, volatility_regime):
        """动态计算止损止盈"""
        try:
            # 基础ATR倍数
            if regime == 'trending':
                base_stop_atr = self.get_param('trend_stop_atr')
                base_tp_atr = 3.0
            else:
                base_stop_atr = self.get_param('grid_stop_atr')
                base_tp_atr = self.get_param('grid_take_profit_atr')
            
            # 根据波动率调整
            volatility_multiplier = 1.0
            if volatility_regime == 'high':
                volatility_multiplier = 1.3  # 高波动率扩大止损
            elif volatility_regime == 'low':
                volatility_multiplier = 0.8  # 低波动率收紧止损
            
            # 根据激进度调整
            aggression_config = self.aggression_controller.get_current_config()
            aggression_multiplier = aggression_config.get('risk_multiplier', 1.0)
            
            final_stop_atr = base_stop_atr * volatility_multiplier * aggression_multiplier
            final_tp_atr = base_tp_atr * volatility_multiplier * aggression_multiplier
            
            # 计算具体价格
            if side == 'LONG':
                stop_loss = price - atr * final_stop_atr
                take_profit = price + atr * final_tp_atr
            else:
                stop_loss = price + atr * final_stop_atr
                take_profit = price - atr * final_tp_atr
            
            LOG.debug(f"{symbol} {side} 动态风控: 止损{final_stop_atr:.2f}ATR, 止盈{final_tp_atr:.2f}ATR")
            
            return stop_loss, take_profit
            
        except Exception as e:
            LOG.error(f"动态止损计算失败: {e}")
            # 回退到固定值
            if side == 'LONG':
                return price * 0.95, price * 1.05
            else:
                return price * 1.05, price * 0.95
    
    def cleanup(self):
        """清理资源"""
        pass

# ========= 增强的自动系统监控 =========
class EnhancedAutoSystemMonitor(AutoSystemMonitor):
    """增强的自动系统监控和恢复"""
    
    def _auto_recover(self):
        """细化自动恢复策略"""
        LOG.info("🔄 执行自动恢复程序")
        
        try:
            # 诊断错误类型
            error_type = self._diagnose_error_type()
            
            if error_type == 'NETWORK':
                self._recover_from_network_error()
            elif error_type == 'API':
                self._recover_from_api_error()
            elif error_type == 'DATA':
                self._recover_from_data_error()
            elif error_type == 'STRATEGY':
                self._recover_from_strategy_error()
            else:
                self._recover_general()
            
            self.bot.consecutive_errors = 0
            LOG.info("✅ 自动恢复完成")
        except Exception as e:
            LOG.error(f'自动恢复失败: {e}')

    def _diagnose_error_type(self):
        """诊断错误类型"""
        try:
            # 测试网络连接
            self.bot.ex.ex.fetch_time()
            return 'STRATEGY'  # 网络正常，可能是策略错误
        except ccxt.NetworkError:
            return 'NETWORK'
        except ccxt.ExchangeError:
            return 'API'
        except (ValueError, KeyError, IndexError):
            return 'DATA'
        except Exception:
            return 'GENERAL'

    def _recover_from_network_error(self):
        """网络错误恢复"""
        LOG.info("🌐 网络错误恢复: 等待重连")
        time.sleep(10)
        # 清理网络相关资源
        self.bot.ex.ex.close()
        # 重新初始化会话
        self.bot.ex.ex = ccxt.gate({
            'apiKey': self.bot.ex.config_manager.get_config_value('apiKey'),
            'secret': self.bot.ex.config_manager.get_config_value('secret'),
            'session': self.bot.ex.SESSION
        })

    def _recover_from_api_error(self):
        """API错误恢复"""
        LOG.info("🔑 API错误恢复: 验证权限")
        # 重新验证API权限
        if not sanity_check_api(self.bot.ex.ex):
            LOG.error("API权限验证失败")
        time.sleep(5)

    def _recover_from_data_error(self):
        """数据错误恢复"""
        LOG.info("📊 数据错误恢复: 清理缓存")
        # 清理数据缓存
        self.bot.last_regime = {}
        # 重新获取数据
        for symbol in self.bot.get_param('symbols'):
            try:
                self.bot.ex.fetch_mtf(symbol)
            except:
                pass
        time.sleep(2)

    def _recover_from_strategy_error(self):
        """策略错误恢复"""
        LOG.info("🎯 策略错误恢复: 重置状态")
        # 重置策略状态
        self.bot.dynamic_params = {
            'base_trend_unit': 150,
            'base_grid_unit': 80,
            'trend_stop_atr': 1.8,
            'grid_stop_atr': 1.2,
            'grid_take_profit_atr': 2.5,
            'rsi_oversold': 28,
            'rsi_overbought': 72,
            'adx_trend_threshold': 22,
            'max_position_ratio': 2.0
        }
        self.bot.weight_system.condition_weights = self.bot.weight_system._initialize_weights()
        time.sleep(1)

    def _recover_general(self):
        """通用恢复"""
        LOG.info("🔄 通用恢复: 重新初始化组件")
        self._cleanup_locks()
        self._reinitialize_components()

# ========= 信号处理器 =========
def signal_handler(signum, frame):
    LOG.info(f'收到信号 {signum}，优雅退出')
    raise KeyboardInterrupt

for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
    signal.signal(sig, signal_handler)

# ========= 主程序 =========
if __name__ == '__main__':
    # 执行时钟同步检查
    check_time_sync()
    
    # 创建并启动机器人
    bot = AITradingBot()
    
    # 替换为增强的监控器
    bot.auto_monitor = EnhancedAutoSystemMonitor(bot)
    
    bot.run()
    
    # 清理资源
    LOG.info('交易机器人已停止')