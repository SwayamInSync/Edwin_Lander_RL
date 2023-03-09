import { makeExplosion } from "./explosion.js";
import { makeConfetti } from "./confetti.js";
import {
  randomBetween,
  randomBool,
  getVectorVelocity,
  getDisplayVelocity,
  getAngleDeltaUpright,
  getAngleDeltaUprightWithSign,
  getDisplayHeight,
  percentProgress,
} from "../helpers/helpers.js";
import {
  scoreLanding,
  scoreCrash,
  landingScoreDescription,
  crashScoreDescription,
} from "../helpers/scoring.js";
import { drawTrajectory } from "./trajectory.js";
import {
  GRAVITY,
  LANDER_WIDTH,
  LANDER_HEIGHT,
  CRASH_VELOCITY,
  CRASH_ANGLE,
} from "../helpers/constants.js";
import { drawLanderGradient } from "./gradient.js";

export const makeLander = (state, onGameEnd, onResetXPos) => {
  const CTX = state.get("CTX");
  const canvasWidth = state.get("canvasWidth");
  const canvasHeight = state.get("canvasHeight") - 30;

  const _thrust = 0.01;
  const _groundedHeight = canvasHeight - LANDER_HEIGHT + LANDER_HEIGHT / 2;

  let _position;
  let _velocity;
  let _rotationVelocity;
  let _angle;
  let _engineOn;
  let _rotatingLeft;
  let _rotatingRight;

  let _crashed;
  let _landed;
  let _flipConfetti;
  let _lastRotation;
  let _lastRotationAngle;
  let _rotationCount;
  let _maxSpeed;
  let _maxHeight;
  let _engineUsed;
  let _engineUsedPreviousFrame;
  let _boostersUsed;
  let _boostersUsedPreviousFrame;
  let _babySoundPlayed;

  const resetProps = () => {
    _position = {
      x: randomBetween(canvasWidth * 0.33, canvasWidth * 0.66),
      y: LANDER_HEIGHT * 2,
    };
    _velocity = {
      x: randomBetween(
        -_thrust * (canvasWidth / 10),
        _thrust * (canvasWidth / 10)
      ),
      y: randomBetween(0, _thrust * (canvasWidth / 10)),
    };
    _rotationVelocity = randomBetween(-0.1, 0.1);
    _angle = randomBetween(Math.PI * 1.5, Math.PI * 2.5);
    _engineOn = false;
    _rotatingLeft = false;
    _rotatingRight = false;

    _crashed = false;
    _landed = false;
    _flipConfetti = [];
    _lastRotation = 1;
    _lastRotationAngle = Math.PI * 2;
    _rotationCount = 0;
    _maxSpeed = 0;
    _maxHeight = _position.y;
    _engineUsed = 0;
    _engineUsedPreviousFrame = false;
    _boostersUsed = 0;
    _boostersUsedPreviousFrame = false;
    _babySoundPlayed = false;
  };
  resetProps();

  const _updateProps = (timeSinceStart) => {
    _position.y = Math.min(_position.y + _velocity.y, _groundedHeight);

    // Is above ground
    if (_position.y < _groundedHeight) {
      // Update ballistic properties
      if (_rotatingRight) _rotationVelocity += 0.01;
      if (_rotatingLeft) _rotationVelocity -= 0.01;

      if (_position.x < 0) {
        _position.x = canvasWidth;
        onResetXPos();
      }
      if (_position.x > canvasWidth) {
        _position.x = 0;
        onResetXPos();
      }

      _position.x += _velocity.x;
      _angle += (Math.PI / 180) * _rotationVelocity;
      _velocity.y += GRAVITY;

      if (_engineOn) {
        _velocity.x += _thrust * Math.sin(_angle);
        _velocity.y -= _thrust * Math.cos(_angle);
      }

      // Log new rotations
      const rotations = Math.floor(_angle / (Math.PI * 2));
      if (
        Math.abs(_angle - _lastRotationAngle) > Math.PI * 2 &&
        (rotations > _lastRotation || rotations < _lastRotation)
      ) {
        _rotationCount++;
        _lastRotation = rotations;
        _lastRotationAngle = _angle;
        _flipConfetti.push(makeConfetti(state, 10, _position, _velocity));
      }

      // Log new max speed and height
      if (_position.y < _maxHeight) _maxHeight = _position.y;

      if (getDisplayVelocity(_velocity) > _maxSpeed) {
        _maxSpeed = getDisplayVelocity(_velocity);
      }

      // Log engine and booster usage
      if (!_engineUsedPreviousFrame && _engineOn) {
        _engineUsed++;
        _engineUsedPreviousFrame = true;
      } else if (_engineUsedPreviousFrame && !_engineOn) {
        _engineUsedPreviousFrame = false;
      }

      if (!_boostersUsedPreviousFrame && (_rotatingLeft || _rotatingRight)) {
        _boostersUsed++;
        _boostersUsedPreviousFrame = true;
      } else if (
        _boostersUsedPreviousFrame &&
        !_rotatingLeft &&
        !_rotatingRight
      ) {
        _boostersUsedPreviousFrame = false;
      }

      // Play easter egg baby sound
      if (getDisplayVelocity(_velocity) > 400 && !_babySoundPlayed) {
        state.get("audioManager").playBaby();
        _babySoundPlayed = true;
      } else if (getDisplayVelocity(_velocity) < 400 && _babySoundPlayed) {
        _babySoundPlayed = false;
      }
    }
    // Just landed or crashed, game over
    else {
      const landed =
        !_landed &&
        getVectorVelocity(_velocity) < CRASH_VELOCITY &&
        getAngleDeltaUpright(_angle) < CRASH_ANGLE;
      const crashed = !_landed && !_crashed;
      const speedInMPH = getDisplayVelocity(_velocity);
      const angleInDeg = getAngleDeltaUpright(_angle);
      const score = landed
        ? scoreLanding(angleInDeg, speedInMPH, _rotationCount)
        : scoreCrash(angleInDeg, speedInMPH, _rotationCount);

      const commonStats = {
        landed,
        score: score.toFixed(1),
        speed: speedInMPH.toFixed(1),
        angle: angleInDeg.toFixed(1),
        durationInSeconds: Math.round(timeSinceStart / 1000),
        rotations: _rotationCount,
        maxSpeed: _maxSpeed.toFixed(1),
        maxHeight: getDisplayHeight(_maxHeight, _groundedHeight),
        engineUsed: _engineUsed,
        boostersUsed: _boostersUsed,
      };

      if (landed) {
        _landed = {
          ...commonStats,
          description: landingScoreDescription(score),
          speedPercent: percentProgress(
            0,
            CRASH_VELOCITY,
            getVectorVelocity(_velocity)
          ),
          anglePercent: percentProgress(0, CRASH_ANGLE, angleInDeg),
          confetti: makeConfetti(state, Math.round(score)),
        };
        _angle = Math.PI * 2;
        _velocity = {
          x: 0,
          y: 0,
        };
        _rotationVelocity = 0;
        onGameEnd(_landed);
      } else if (crashed) {
        _crashed = {
          ...commonStats,
          description: crashScoreDescription(score),
          speedPercent: percentProgress(
            0,
            CRASH_VELOCITY,
            getVectorVelocity(_velocity)
          ),
          anglePercent: percentProgress(0, CRASH_ANGLE, angleInDeg),
          explosion: makeExplosion(state, _position, _velocity, _angle),
        };
        onGameEnd(_crashed);
      }
    }
  };

  const _drawSideBySideStats = () => {
    const textWidth = CTX.measureText("100.0 MPH").width + 2;
    const staticPosition = getVectorVelocity(_velocity) > 10;
    const xPosBasis = staticPosition
      ? 8
      : Math.min(_position.x + LANDER_WIDTH * 2, canvasWidth - textWidth);
    const yPosBasis = Math.max(_position.y, 30);
    const lineHeight = 14;

    let v = getDisplayVelocity(_velocity).toFixed(1);
    let a = getAngleDeltaUprightWithSign(_angle).toFixed(1);
    let h = getDisplayHeight(_position.y, _groundedHeight);

    CTX.save();
    CTX.fillStyle =
      getVectorVelocity(_velocity) > CRASH_VELOCITY
        ? "rgb(255, 0, 0)"
        : "rgb(0, 255, 0)";
    CTX.fillText(
      `${getDisplayVelocity(_velocity).toFixed(1)} MPH`,
      xPosBasis,
      yPosBasis - lineHeight
    );
    CTX.fillStyle =
      getAngleDeltaUpright(_angle) > CRASH_ANGLE
        ? "rgb(255, 0, 0)"
        : "rgb(0, 255, 0)";
    CTX.fillText(
      `${getAngleDeltaUprightWithSign(_angle).toFixed(1)}°`,
      xPosBasis,
      yPosBasis
    );
    CTX.fillStyle = "#ffffff";
    CTX.fillText(
      `${getDisplayHeight(_position.y, _groundedHeight)} FT`,
      xPosBasis,
      yPosBasis + lineHeight
    );
    CTX.restore();
    _drawSideBySideStats;
    // return { v, a, h };
    const element = document.querySelector("#stats");
    let sp = document.querySelector("#stats span");
    if (sp === null) {
      sp = document.createElement("span");
      sp.innerHTML = `<div style="margin: auto;width: 50%;padding: 10px;">${v}, ${a}, ${h}</div>`;
      element.appendChild(sp);
    } else {
      sp.innerHTML = `<div style="margin: auto;width: 50%;padding: 10px;">${v}, ${a}, ${h}</div>`;
    }
  };

  const _drawLander = () => {
    // Move to top left of the lander and then rotate at that origin
    CTX.save();
    CTX.fillStyle = drawLanderGradient(CTX);
    CTX.translate(_position.x, _position.y);
    CTX.rotate(_angle);

    // Draw the lander
    //
    // We want the center of rotation to be in the center of the bottom
    // rectangle, excluding the tip of the lander. To accomplish this, the
    // lander is drawn offset to the top and left of _position.x and y.
    // The tip is also drawn offset to the top of that so that the lander
    // is a bit taller than LANDER_HEIGHT.
    //
    //                                      /\
    //                                     /  \
    // Start at top left of this segment → |  |
    // and work clockwise.                 |__|
    CTX.beginPath();
    CTX.moveTo(-LANDER_WIDTH / 2, -LANDER_HEIGHT / 2);
    CTX.lineTo(0, -LANDER_HEIGHT);
    CTX.lineTo(LANDER_WIDTH / 2, -LANDER_HEIGHT / 2);
    CTX.lineTo(LANDER_WIDTH / 2, LANDER_HEIGHT / 2);
    CTX.lineTo(-LANDER_WIDTH / 2, LANDER_HEIGHT / 2);
    CTX.closePath();
    CTX.fill();

    // Translate to the top-left corner of the lander so engine and booster
    // flames can be drawn from 0, 0
    CTX.translate(-LANDER_WIDTH / 2, -LANDER_HEIGHT / 2);

    if (_engineOn || _rotatingLeft || _rotatingRight) {
      CTX.fillStyle = randomBool() ? "#415B8C" : "#F3AFA3";
    }

    // Main engine flame
    if (_engineOn) {
      const _flameHeight = randomBetween(10, 50);
      const _flameMargin = 3;
      CTX.beginPath();
      CTX.moveTo(_flameMargin, LANDER_HEIGHT);
      CTX.lineTo(LANDER_WIDTH - _flameMargin, LANDER_HEIGHT);
      CTX.lineTo(LANDER_WIDTH / 2, LANDER_HEIGHT + _flameHeight);
      CTX.closePath();
      CTX.fill();
    }

    const _boosterLength = randomBetween(5, 25);
    // Right booster flame
    if (_rotatingLeft) {
      CTX.beginPath();
      CTX.moveTo(LANDER_WIDTH, 0);
      CTX.lineTo(LANDER_WIDTH + _boosterLength, LANDER_HEIGHT * 0.05);
      CTX.lineTo(LANDER_WIDTH, LANDER_HEIGHT * 0.1);
      CTX.closePath();
      CTX.fill();
    }

    // Left booster flame
    if (_rotatingRight) {
      CTX.beginPath();
      CTX.moveTo(0, 0);
      CTX.lineTo(-_boosterLength, LANDER_HEIGHT * 0.05);
      CTX.lineTo(0, LANDER_HEIGHT * 0.1);
      CTX.closePath();
      CTX.fill();
    }

    CTX.restore();
  };

  const draw = (timeSinceStart) => {
    _updateProps(timeSinceStart);

    if (!_engineOn && !_rotatingLeft && !_rotatingRight) {
      drawTrajectory(
        state,
        _position,
        _angle,
        _velocity,
        _rotationVelocity,
        _groundedHeight
      );
    }

    if (_flipConfetti.length > 0) _flipConfetti.forEach((c) => c.draw());

    if (_landed) _landed.confetti.draw();

    // Don't draw the lander after crashing
    if (_crashed) _crashed.explosion.draw();
    else _drawLander();

    // Draw speed and angle text beside lander, even after crashing
    _drawSideBySideStats();
  };

  return {
    draw,
    resetProps,
    engineOn: () => (_engineOn = true),
    engineOff: () => (_engineOn = false),
    rotateLeft: () => (_rotatingLeft = true),
    rotateRight: () => (_rotatingRight = true),
    stopLeftRotation: () => (_rotatingLeft = false),
    stopRightRotation: () => (_rotatingRight = false),
  };
};
