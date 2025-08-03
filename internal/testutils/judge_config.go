package testutils

import (
	"encoding/json"
	"fmt"
	"os"
)

// JudgeConfig defines configuration parameters for individual judge behavior.
// These parameters control how accurately judges identify correct answers,
// how much noise they add to scores, and their error characteristics.
type JudgeConfig struct {
	// CorrectAnswerAccuracy is the probability (0.0-1.0) that the judge
	// correctly identifies and scores a correct answer highly.
	// Default: 0.85 (85% accuracy)
	CorrectAnswerAccuracy float64 `json:"correct_answer_accuracy"`

	// IncorrectAnswerAccuracy is the probability (0.0-1.0) that the judge
	// correctly identifies and scores an incorrect answer lowly.
	// Default: 0.80 (80% accuracy)
	IncorrectAnswerAccuracy float64 `json:"incorrect_answer_accuracy"`

	// NoiseFactor controls the amount of random variation added to scores.
	// Higher values create more variance in judge decisions.
	// Default: 0.10 (10% noise)
	NoiseFactor float64 `json:"noise_factor"`

	// LargeErrorProbability is the probability (0.0-1.0) of making a
	// significantly larger error than normal.
	// Default: 0.05 (5% chance)
	LargeErrorProbability float64 `json:"large_error_probability"`

	// LargeErrorMultiplier is how much to multiply the noise by when
	// a large error occurs.
	// Default: 1.5
	LargeErrorMultiplier float64 `json:"large_error_multiplier"`

	// PersonalityStrength controls how much the judge's personality
	// affects their scoring (0.0 = no effect, 1.0 = maximum effect).
	// Default: 0.5
	PersonalityStrength float64 `json:"personality_strength"`
}

// DefaultJudgeConfig returns a JudgeConfig with sensible default values.
func DefaultJudgeConfig() JudgeConfig {
	return JudgeConfig{
		CorrectAnswerAccuracy:   0.85,
		IncorrectAnswerAccuracy: 0.80,
		NoiseFactor:             0.10,
		LargeErrorProbability:   0.05,
		LargeErrorMultiplier:    1.5,
		PersonalityStrength:     0.5,
	}
}

// Validate checks that all configuration values are within valid ranges.
func (c JudgeConfig) Validate() error {
	if c.CorrectAnswerAccuracy < 0.0 || c.CorrectAnswerAccuracy > 1.0 {
		return fmt.Errorf("correct_answer_accuracy must be between 0.0 and 1.0, got %f", c.CorrectAnswerAccuracy)
	}
	if c.IncorrectAnswerAccuracy < 0.0 || c.IncorrectAnswerAccuracy > 1.0 {
		return fmt.Errorf("incorrect_answer_accuracy must be between 0.0 and 1.0, got %f", c.IncorrectAnswerAccuracy)
	}
	if c.NoiseFactor < 0.0 || c.NoiseFactor > 1.0 {
		return fmt.Errorf("noise_factor must be between 0.0 and 1.0, got %f", c.NoiseFactor)
	}
	if c.LargeErrorProbability < 0.0 || c.LargeErrorProbability > 1.0 {
		return fmt.Errorf("large_error_probability must be between 0.0 and 1.0, got %f", c.LargeErrorProbability)
	}
	if c.LargeErrorMultiplier < 1.0 || c.LargeErrorMultiplier > 10.0 {
		return fmt.Errorf("large_error_multiplier must be between 1.0 and 10.0, got %f", c.LargeErrorMultiplier)
	}
	if c.PersonalityStrength < 0.0 || c.PersonalityStrength > 1.0 {
		return fmt.Errorf("personality_strength must be between 0.0 and 1.0, got %f", c.PersonalityStrength)
	}
	return nil
}

// EnsembleConfig defines configuration for ensemble behavior and judge interactions.
// These parameters control how judges in an ensemble might influence each other
// and exhibit correlated errors.
type EnsembleConfig struct {
	// ErrorCorrelation is the probability (0.0-1.0) that judges make
	// similar errors. Higher values mean judges are more likely to
	// make the same mistakes.
	// Default: 0.0 (independent errors)
	ErrorCorrelation float64 `json:"error_correlation"`

	// SharedErrorSeed is used to generate correlated errors across judges.
	// When non-zero, judges will share some error patterns.
	// Default: 0 (no shared seed)
	SharedErrorSeed int64 `json:"shared_error_seed"`

	// BiasAmplification controls how much biases are amplified when
	// multiple judges share similar biases (0.0-1.0).
	// Default: 0.0 (no amplification)
	BiasAmplification float64 `json:"bias_amplification"`

	// ConfidenceThreshold is the minimum confidence level required
	// for a judge's score to be included in ensemble decisions.
	// Default: 0.0 (all scores included)
	ConfidenceThreshold float64 `json:"confidence_threshold"`

	// JudgeConfigs maps judge names to their individual configurations.
	// This allows fine-tuning each judge in the ensemble.
	JudgeConfigs map[string]JudgeConfig `json:"judge_configs"`
}

// DefaultEnsembleConfig returns an EnsembleConfig with sensible default values.
func DefaultEnsembleConfig() EnsembleConfig {
	return EnsembleConfig{
		ErrorCorrelation:    0.0,
		SharedErrorSeed:     0,
		BiasAmplification:   0.0,
		ConfidenceThreshold: 0.0,
		JudgeConfigs:        make(map[string]JudgeConfig),
	}
}

// Validate checks that all configuration values are within valid ranges.
func (c EnsembleConfig) Validate() error {
	if c.ErrorCorrelation < 0.0 || c.ErrorCorrelation > 1.0 {
		return fmt.Errorf("error_correlation must be between 0.0 and 1.0, got %f", c.ErrorCorrelation)
	}
	if c.BiasAmplification < 0.0 || c.BiasAmplification > 1.0 {
		return fmt.Errorf("bias_amplification must be between 0.0 and 1.0, got %f", c.BiasAmplification)
	}
	if c.ConfidenceThreshold < 0.0 || c.ConfidenceThreshold > 1.0 {
		return fmt.Errorf("confidence_threshold must be between 0.0 and 1.0, got %f", c.ConfidenceThreshold)
	}

	// Validate individual judge configs
	for name, config := range c.JudgeConfigs {
		if err := config.Validate(); err != nil {
			return fmt.Errorf("judge config %s: %w", name, err)
		}
	}

	return nil
}

// LoadJudgeConfig loads a JudgeConfig from a JSON file.
func LoadJudgeConfig(path string) (JudgeConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return JudgeConfig{}, fmt.Errorf("failed to read config file: %w", err)
	}

	var config JudgeConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return JudgeConfig{}, fmt.Errorf("failed to parse config JSON: %w", err)
	}

	if err := config.Validate(); err != nil {
		return JudgeConfig{}, fmt.Errorf("config validation failed: %w", err)
	}

	return config, nil
}

// LoadEnsembleConfig loads an EnsembleConfig from a JSON file.
func LoadEnsembleConfig(path string) (EnsembleConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return EnsembleConfig{}, fmt.Errorf("failed to read config file: %w", err)
	}

	var config EnsembleConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return EnsembleConfig{}, fmt.Errorf("failed to parse config JSON: %w", err)
	}

	if err := config.Validate(); err != nil {
		return EnsembleConfig{}, fmt.Errorf("config validation failed: %w", err)
	}

	return config, nil
}
