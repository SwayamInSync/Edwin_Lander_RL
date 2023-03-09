import {
  CRASH_VELOCITY,
  CRASH_ANGLE,
  VELOCITY_MULTIPLIER,
} from "./constants.js";
import { progress } from "./helpers.js";

export const landingScoreDescription = (score) => {
  const description = "PERFECT LANDING";

  return description;
};

export const crashScoreDescription = (score) => {
  const description = "CRASH";
  return description;
};

// Perfect land:
// angle: 0
// speed: 1
// rotations: bonus, higher better
//
// Worst possible landing:
// angle: 11
// speed: 12
// rotations: bonus, higher better
export const scoreLanding = (angle, speed, rotations) => {
  const bestPossibleCombo = 1;
  const worstPossibleCombo = CRASH_ANGLE + CRASH_VELOCITY * VELOCITY_MULTIPLIER;
  const combinedStats = angle + speed;
  const score =
    progress(worstPossibleCombo, bestPossibleCombo, combinedStats) * 100;
  return score;
};

// Least bad possible crash:
// angle: 0
// speed: 9
// rotations: bonus, higher better
//
// Also least bad possible crash:
// angle: 11
// speed: 1
// rotations: bonus, higher better
//
// Expected best possible crash
// speed: 1000
// angle: 180
// rotations: bonus, higher better
export const scoreCrash = (angle, speed, rotations) => {
  const worstPossibleCombo = Math.min(CRASH_VELOCITY, CRASH_ANGLE);
  const bestPossibleCombo = 900;
  const combinedStats = angle + speed;
  const score =
    progress(worstPossibleCombo, bestPossibleCombo, combinedStats) * 100;

  return score;
};
