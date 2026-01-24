"""
Regime Statistics Replay Driver (Phase v2.0).

Replay regime classification across multiple historical days and
generate statistical validation reports.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path
import logging

from analytics.regime_statistics import RegimeStatistics


class RegimeStatisticsReplay:
    """Replay regime statistics across configurable date ranges."""
    
    def __init__(self, output_dir: str = 'logs/regime'):
        """Initialize regime statistics replay."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logger()
        self.stats = RegimeStatistics(logger=self.logger)
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger('RegimeStatsReplay')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_dir / f'regime_stats_replay_{timestamp}.log'
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _create_synthetic_day(
        self,
        day_type: str,
        base_date: datetime,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Create synthetic market data for a specific day type.
        
        Args:
            day_type: 'trend', 'range', or 'reversal'
            base_date: Base date for the synthetic day
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with OHLCV and VWAP data
        """
        if seed is not None:
            np.random.seed(seed)
        
        timestamps = pd.date_range(base_date, periods=480, freq='1min')
        
        prices = [4500.0]
        volumes = [np.random.randint(1000, 2500)]  # Add initial volume
        
        for i in range(1, len(timestamps)):
            if day_type == 'trend':
                # Strong directional move with occasional retraces
                drift = 0.0003
                if i % 120 == 0:
                    drift *= -0.5  # occasional pullback
                noise = np.random.normal(0, 3)
                price = prices[-1] + (drift * prices[-1]) + noise
                volume = np.random.randint(1500, 3000)
            
            elif day_type == 'range':
                # Oscillation around center
                center_price = 4500.0
                oscillation = 20 * np.sin(2 * np.pi * i / 480)
                noise = np.random.normal(0, 3)
                price = center_price + oscillation + noise
                volume = np.random.randint(1000, 2500)
            
            elif day_type == 'reversal':
                # Strong up then sharp down (or vice versa)
                if i < 240:
                    drift = 0.0004
                else:
                    drift = -0.0006
                noise = np.random.normal(0, 3)
                price = prices[-1] + (drift * prices[-1]) + noise
                volume = np.random.randint(1200, 3500)
            
            else:  # choppy/range-like
                drift = np.random.normal(0, 0.0001)
                noise = np.random.normal(0, 2)
                price = prices[-1] + (drift * prices[-1]) + noise
                volume = np.random.randint(800, 2000)
            
            prices.append(max(price, 4400))  # Prevent negative prices
            volumes.append(volume)
        
        prices = np.array(prices)
        
        # Calculate VWAP
        vwap = prices.copy()
        cumulative_tp_volume = 0
        cumulative_volume = 0
        for i in range(len(prices)):
            typical_price = (prices[i] + prices[i] + prices[i]) / 3  # Simplified
            cumulative_tp_volume += typical_price * volumes[i]
            cumulative_volume += volumes[i]
            vwap[i] = cumulative_tp_volume / cumulative_volume if cumulative_volume > 0 else prices[i]
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices + np.random.uniform(0.5, 2, len(prices)),
            'low': prices - np.random.uniform(0.5, 2, len(prices)),
            'close': prices + np.random.normal(0, 1, len(prices)),
            'volume': volumes,
            'vwap': vwap,
            'session': 'RTH',
            'initiative': np.random.rand(len(prices)) > 0.7,
            'stop_run': np.random.rand(len(prices)) > 0.9,
            'flow_signals': [{}] * len(prices),
        })
        
        return data
    
    def replay_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        day_type_distribution: Optional[dict] = None
    ) -> None:
        """
        Replay regime classification across a date range.
        
        Args:
            start_date: Start date for replay
            end_date: End date for replay
            day_type_distribution: Dict with day type frequencies (e.g., {'trend': 0.3, 'range': 0.5})
        """
        if day_type_distribution is None:
            day_type_distribution = {'trend': 0.25, 'range': 0.5, 'reversal': 0.25}
        
        current_date = start_date
        day_count = 0
        day_types = list(day_type_distribution.keys())
        weights = list(day_type_distribution.values())
        
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info("REGIME STATISTICS REPLAY")
        self.logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")
        self.logger.info(f"Day Type Distribution: {day_type_distribution}")
        self.logger.info(f"{'=' * 80}\n")
        
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            
            # Pick day type based on distribution
            day_type = np.random.choice(day_types, p=weights)
            
            # Generate synthetic day
            day_data = self._create_synthetic_day(day_type, current_date, seed=int(current_date.timestamp()))
            
            # Analyze day
            day_stat = self.stats.analyze_day(day_data, current_date, symbol='ES')
            
            day_count += 1
            self.logger.info(
                f"[{day_count:3d}] {current_date.date()} ({day_type:8}) -> "
                f"{day_stat.dominant_regime:10} ({day_stat.dominant_regime_pct:5.1%}) | "
                f"Transitions: {day_stat.transition_count:2d}"
            )
            
            current_date += timedelta(days=1)
        
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"Total days replayed: {day_count}")
        self.logger.info(f"{'=' * 80}\n")
    
    def generate_report(self) -> str:
        """Generate and save comprehensive statistics report."""
        report = self.stats.generate_summary_report()
        
        # Save report to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f'regime_stats_{timestamp}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Report saved to: {report_file}")
        
        return report
    
    def run_statistical_validation(
        self,
        num_days: int = 60,
        day_type_distribution: Optional[dict] = None
    ) -> str:
        """
        Run complete statistical validation.
        
        Args:
            num_days: Number of days to analyze
            day_type_distribution: Day type frequency distribution
            
        Returns:
            Generated report as string
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days * 7 // 5)  # Account for weekends
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PHASE v2.0: REGIME STATISTICS VALIDATION")
        self.logger.info("=" * 80)
        
        # Run replay
        self.replay_date_range(start_date, end_date, day_type_distribution)
        
        # Generate report
        report = self.generate_report()
        
        # Print to console
        self.logger.info("\n" + report)
        
        return report
    
    def print_summary(self) -> None:
        """Print summary of analysis to console."""
        if not self.stats.day_stats:
            self.logger.info("No days analyzed yet")
            return
        
        aggregate = self.stats.aggregate_statistics()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("QUICK SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Days: {aggregate.total_days}")
        self.logger.info(f"Total Bars: {aggregate.total_bars}")
        self.logger.info(f"\nRegime Distribution:")
        for regime, freq_data in sorted(aggregate.regime_frequency.items()):
            self.logger.info(f"  {regime:12} {freq_data['pct_of_total_bars']:6.1%} "
                           f"({freq_data['bars']:4d} bars)")
        self.logger.info(f"\nDay Type Distribution:")
        for day_type, count in sorted(aggregate.day_type_distribution.items()):
            pct = count / aggregate.total_days * 100
            self.logger.info(f"  {day_type:12} {count:3d} days ({pct:5.1f}%)")


def main():
    """Main entry point for regime statistics replay."""
    import sys
    
    # Parse command line arguments
    num_days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    
    # Run validation
    replay = RegimeStatisticsReplay(output_dir='logs/regime')
    replay.run_statistical_validation(num_days=num_days)
    replay.print_summary()


if __name__ == '__main__':
    main()
