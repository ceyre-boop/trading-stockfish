"""
Test Suite for ExperimentRunner - analytics/experiment_runner.py

Tests cover:
  - ExperimentConfig creation and validation
  - Parameter set generation
  - Experiment execution
  - Results collection and storage
  - Comparison report generation
  - Export functionality
  - Configuration loading from YAML

Author: Trading-Stockfish Tests
Version: 1.0.0
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
import yaml

from analytics.experiment_runner import (
    ExperimentRunner, ExperimentConfig, ParameterSet, ExperimentResult,
    ExperimentStatus, create_default_experiment_config
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_config():
    """Create sample experiment configuration."""
    return ExperimentConfig(
        name="test_experiment",
        description="Test experiment for unit testing",
        macro_weight_range=(0.0, 1.0),
        macro_weight_steps=2,
        volatility_threshold_range=(0.01, 0.05),
        volatility_threshold_steps=2,
        reversal_threshold_range=(0.3, 0.7),
        reversal_threshold_steps=2,
        cooldown_range=(1, 5),
        cooldown_steps=2,
        fomc_trade_bans=[False, True],
        symbols=['EURUSD', 'GBPUSD'],
        start_date="2023-01-01",
        end_date="2024-12-31",
        walkforward_periods=4,
        output_dir="experiments",
        experiment_name="test_experiment",
        verbose=False,
    )


@pytest.fixture
def experiment_runner(sample_config):
    """Create ExperimentRunner instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_config.output_dir = tmpdir
        runner = ExperimentRunner(sample_config)
        yield runner


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestExperimentConfig:
    """Test ExperimentConfig creation and handling."""
    
    def test_config_creation(self, sample_config):
        """Test creating experiment configuration."""
        assert sample_config.name == "test_experiment"
        assert sample_config.macro_weight_range == (0.0, 1.0)
        assert sample_config.macro_weight_steps == 2
        assert len(sample_config.symbols) == 2
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = create_default_experiment_config()
        
        assert config.name == "default_experiment"
        assert config.macro_weight_steps > 0
        assert len(config.symbols) > 0
    
    def test_config_to_dict(self, sample_config):
        """Test config conversion to dict."""
        config_dict = sample_config.to_dict()
        
        assert 'name' in config_dict
        assert 'macro_weight_range' in config_dict
        assert config_dict['name'] == "test_experiment"
        assert config_dict['macro_weight_range'] == (0.0, 1.0)
    
    def test_parameter_set_creation(self):
        """Test creating parameter sets."""
        params = {
            'macro_weight': 0.5,
            'volatility_threshold': 0.02,
            'reversal_threshold': 0.5,
            'cooldown': 5,
            'fomc_trade_ban': True,
        }
        
        param_set = ParameterSet(
            config_id='test_001',
            parameters=params
        )
        
        assert param_set.config_id == 'test_001'
        assert param_set.parameters['macro_weight'] == 0.5
        assert param_set.parameters['fomc_trade_ban'] is True
    
    def test_parameter_set_to_dict(self):
        """Test parameter set conversion."""
        params = {'macro_weight': 0.5}
        param_set = ParameterSet(config_id='test_001', parameters=params)
        
        param_dict = param_set.to_dict()
        assert 'config_id' in param_dict
        assert 'parameters' in param_dict


# =============================================================================
# PARAMETER SET GENERATION TESTS
# =============================================================================

class TestParameterSetGeneration:
    """Test parameter set generation."""
    
    def test_parameter_set_generation_basic(self, experiment_runner):
        """Test basic parameter set generation."""
        param_sets = experiment_runner.generate_parameter_sets()
        
        # Should have combinations of:
        # 2 macro weights * 2 volatility * 2 reversal * 2 cooldown * 2 fomc = 32
        expected_count = 2 * 2 * 2 * 2 * 2
        assert len(param_sets) == expected_count
    
    def test_parameter_set_uniqueness(self, experiment_runner):
        """Test all parameter sets are unique."""
        param_sets = experiment_runner.generate_parameter_sets()
        
        config_ids = [p.config_id for p in param_sets]
        assert len(config_ids) == len(set(config_ids))
    
    def test_parameter_set_completeness(self, experiment_runner):
        """Test parameter sets have all required parameters."""
        param_sets = experiment_runner.generate_parameter_sets()
        
        required_keys = {
            'macro_weight',
            'volatility_threshold',
            'reversal_threshold',
            'cooldown',
            'fomc_trade_ban',
        }
        
        for param_set in param_sets:
            assert set(param_set.parameters.keys()) == required_keys
    
    def test_parameter_set_value_ranges(self, experiment_runner):
        """Test parameter values are within specified ranges."""
        config = experiment_runner.config
        param_sets = experiment_runner.generate_parameter_sets()
        
        for param_set in param_sets:
            params = param_set.parameters
            
            # Check macro weight
            assert config.macro_weight_range[0] <= params['macro_weight'] <= config.macro_weight_range[1]
            
            # Check volatility threshold
            assert config.volatility_threshold_range[0] <= params['volatility_threshold'] <= config.volatility_threshold_range[1]
            
            # Check reversal threshold
            assert config.reversal_threshold_range[0] <= params['reversal_threshold'] <= config.reversal_threshold_range[1]
            
            # Check cooldown
            assert config.cooldown_range[0] <= params['cooldown'] <= config.cooldown_range[1]
            
            # Check FOMC ban
            assert params['fomc_trade_ban'] in config.fomc_trade_bans


# =============================================================================
# EXPERIMENT EXECUTION TESTS
# =============================================================================

class TestExperimentExecution:
    """Test experiment execution."""
    
    def test_single_experiment_run(self, experiment_runner):
        """Test running a single experiment."""
        param_sets = experiment_runner.generate_parameter_sets()
        param_set = param_sets[0]
        
        result = experiment_runner._run_single_experiment(param_set)
        
        assert result.config_id == param_set.config_id
        assert result.status == ExperimentStatus.COMPLETED.value
        assert result.pnl != 0.0 or result.sharpe_ratio > 0
    
    def test_full_experiment_run(self, experiment_runner):
        """Test running all experiments."""
        param_sets = experiment_runner.generate_parameter_sets()
        
        # Use small subset for testing
        experiment_runner.parameter_sets = param_sets[:2]
        
        results = experiment_runner.run_experiments()
        
        assert len(results) == 2
        assert all(r.status == ExperimentStatus.COMPLETED.value for r in results)
    
    def test_experiment_result_creation(self):
        """Test creating experiment results."""
        result = ExperimentResult(
            config_id='test_001',
            status=ExperimentStatus.COMPLETED.value,
            pnl=1000.0,
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            win_rate=0.55,
            trades_count=100,
            elo_rating=1700.0,
        )
        
        assert result.config_id == 'test_001'
        assert result.pnl == 1000.0
        assert result.elo_rating == 1700.0
    
    def test_result_to_dict(self):
        """Test result conversion to dict."""
        result = ExperimentResult(
            config_id='test_001',
            status=ExperimentStatus.COMPLETED.value,
            pnl=1000.0,
            sharpe_ratio=1.5,
        )
        
        result_dict = result.to_dict()
        
        assert 'config_id' in result_dict
        assert 'status' in result_dict
        assert 'metrics' in result_dict
        assert result_dict['metrics']['pnl'] == 1000.0


# =============================================================================
# COMPARISON AND REPORTING TESTS
# =============================================================================

class TestComparisonAndReporting:
    """Test comparison report generation."""
    
    def test_comparison_report_generation(self, experiment_runner):
        """Test generating comparison report."""
        # Run small experiment
        param_sets = experiment_runner.generate_parameter_sets()[:3]
        experiment_runner.parameter_sets = param_sets
        experiment_runner.run_experiments()
        
        report = experiment_runner.compare_results()
        
        assert 'total_experiments' in report
        assert 'statistics' in report
        assert 'top_10' in report
        assert report['total_experiments'] == 3
    
    def test_comparison_statistics(self, experiment_runner):
        """Test comparison statistics calculation."""
        # Create sample results
        for i in range(5):
            result = ExperimentResult(
                config_id=f'config_{i}',
                status=ExperimentStatus.COMPLETED.value,
                pnl=1000.0 + i * 100,
                sharpe_ratio=1.0 + i * 0.1,
                max_drawdown=0.1 + i * 0.02,
                elo_rating=1600.0 + i * 50,
            )
            experiment_runner.results.append(result)
        
        report = experiment_runner.compare_results()
        
        # Check statistics
        assert 'pnl' in report['statistics']
        assert 'sharpe_ratio' in report['statistics']
        assert 'max_drawdown' in report['statistics']
        assert 'elo_rating' in report['statistics']
        
        # Check that stats are computed
        pnl_stats = report['statistics']['pnl']
        assert pnl_stats['min'] < pnl_stats['max']
        assert pnl_stats['mean'] > 0
    
    def test_top_results_ranking(self, experiment_runner):
        """Test top results are ranked correctly."""
        # Create results with different ELO ratings
        for i, elo in enumerate([1500, 1700, 1600, 1800, 1650]):
            result = ExperimentResult(
                config_id=f'config_{i}',
                status=ExperimentStatus.COMPLETED.value,
                elo_rating=float(elo),
                pnl=1000.0,
            )
            experiment_runner.results.append(result)
        
        report = experiment_runner.compare_results()
        top_10 = report['top_10']
        
        # Top result should have highest ELO
        assert top_10[0]['elo_rating'] == 1800.0
        assert top_10[1]['elo_rating'] == 1700.0


# =============================================================================
# EXPORT TESTS
# =============================================================================

class TestExportFunctionality:
    """Test export functionality."""
    
    def test_export_results(self, experiment_runner):
        """Test exporting results."""
        # Run small experiment
        param_sets = experiment_runner.generate_parameter_sets()[:2]
        experiment_runner.parameter_sets = param_sets
        experiment_runner.run_experiments()
        experiment_runner.compare_results()
        
        exported = experiment_runner.export_results()
        
        assert 'config' in exported
        assert 'results' in exported
        assert 'comparison' in exported
        assert 'summary' in exported
        
        for path in exported.values():
            assert path.exists()
    
    def test_exported_config_file(self, experiment_runner):
        """Test exported config file."""
        param_sets = experiment_runner.generate_parameter_sets()[:1]
        experiment_runner.parameter_sets = param_sets
        experiment_runner.run_experiments()
        
        exported = experiment_runner.export_results()
        config_file = exported['config']
        
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        assert 'name' in config_data
        assert 'macro_weight_range' in config_data
    
    def test_exported_results_file(self, experiment_runner):
        """Test exported results file."""
        param_sets = experiment_runner.generate_parameter_sets()[:1]
        experiment_runner.parameter_sets = param_sets
        experiment_runner.run_experiments()
        
        exported = experiment_runner.export_results()
        results_file = exported['results']
        
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        assert isinstance(results_data, list)
        assert len(results_data) >= 1
        assert 'config_id' in results_data[0]
    
    def test_summary_markdown_creation(self, experiment_runner):
        """Test summary markdown is created."""
        param_sets = experiment_runner.generate_parameter_sets()[:2]
        experiment_runner.parameter_sets = param_sets
        experiment_runner.run_experiments()
        experiment_runner.compare_results()
        
        exported = experiment_runner.export_results()
        summary_file = exported['summary']
        
        assert summary_file.exists()
        
        with open(summary_file, 'r') as f:
            content = f.read()
        
        assert '# Experiment Summary' in content
        assert experiment_runner.config.name in content


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestExperimentRunnerIntegration:
    """Integration tests for experiment runner."""
    
    def test_full_workflow(self, sample_config):
        """Test complete experiment workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_config.output_dir = tmpdir
            sample_config.macro_weight_steps = 2  # Keep small
            
            runner = ExperimentRunner(sample_config)
            
            # Generate parameters
            param_sets = runner.generate_parameter_sets()
            assert len(param_sets) > 0
            
            # Run experiments
            runner.parameter_sets = param_sets[:3]  # Use subset
            results = runner.run_experiments()
            assert len(results) == 3
            
            # Compare results
            report = runner.compare_results()
            assert 'statistics' in report
            
            # Export results
            exported = runner.export_results()
            assert len(exported) == 4
    
    def test_multiple_runners_isolated(self, sample_config):
        """Test multiple runners don't interfere."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                config1 = sample_config
                config1.output_dir = tmpdir1
                
                config2 = ExperimentConfig(
                    name="test_exp_2",
                    macro_weight_steps=2,
                    output_dir=tmpdir2,
                    experiment_name="test_exp_2",
                )
                
                runner1 = ExperimentRunner(config1)
                runner2 = ExperimentRunner(config2)
                
                sets1 = runner1.generate_parameter_sets()
                sets2 = runner2.generate_parameter_sets()
                
                # Results should be independent
                assert len(sets1) > 0
                assert len(sets2) > 0


# =============================================================================
# YAML CONFIGURATION TESTS
# =============================================================================

class TestYAMLConfiguration:
    """Test YAML configuration loading."""
    
    def test_config_from_yaml(self):
        """Test loading config from YAML."""
        # Create sample YAML
        config_data = {
            'name': 'yaml_test',
            'description': 'Test YAML config',
            'macro_weight_range': [0.0, 1.0],
            'macro_weight_steps': 3,
            'volatility_threshold_range': [0.01, 0.05],
            'volatility_threshold_steps': 2,
            'reversal_threshold_range': [0.3, 0.7],
            'reversal_threshold_steps': 2,
            'cooldown_range': [1, 5],
            'cooldown_steps': 2,
            'fomc_trade_bans': [False, True],
            'symbols': ['EURUSD'],
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'walkforward_periods': 4,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yaml_file = f.name
        
        try:
            config = ExperimentConfig.from_yaml(Path(yaml_file))
            
            assert config.name == 'yaml_test'
            assert config.macro_weight_range == (0.0, 1.0)
            assert config.symbols == ['EURUSD']
        finally:
            Path(yaml_file).unlink()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_parameter_sweep(self):
        """Test with minimal parameter sweep."""
        config = ExperimentConfig(
            name="minimal",
            macro_weight_steps=1,
            volatility_threshold_steps=1,
            reversal_threshold_steps=1,
            cooldown_steps=1,
            fomc_trade_bans=[False],
            output_dir="experiments",
            experiment_name="minimal",
        )
        
        runner = ExperimentRunner(config)
        param_sets = runner.generate_parameter_sets()
        
        # Should have 1 * 1 * 1 * 1 * 1 = 1 parameter set
        assert len(param_sets) == 1
    
    def test_empty_results_comparison(self):
        """Test comparison with no results."""
        config = create_default_experiment_config()
        runner = ExperimentRunner(config)
        
        report = runner.compare_results()
        
        # Should return empty report
        assert report == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
