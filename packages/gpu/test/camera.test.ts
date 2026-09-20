// The quaternion orbit camera: its view matrix agrees with lookAt (z up) for an upright camera, the helpers are
// consistent, and orbiting over a pole stays a rotation (no gimbal lock: the camera comes back upside down and
// keeps going).

import { describe, expect, it } from "vitest";
import { type Camera3D, type Quat, cameraAxes, cameraEye, cameraMatrices, lookAt, quatAxisAngle, quatLook, quatMul, quatNormalize, quatRotate } from "../src/render3d";

const close = (a: ArrayLike<number>, b: ArrayLike<number>, eps = 1e-6) => { expect(a.length).toBe(b.length); for (let i = 0; i < a.length; i++) expect(Math.abs(a[i]! - b[i]!)).toBeLessThan(eps); };
const fromAngles = (yaw: number, pitch: number) => [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)];

describe("quaternion camera", () => {
  it("quatLook reproduces lookAt for an upright camera", () => {
    for (const [yaw, pitch] of [[-0.9, 0.55], [0, 0], [2.5, -1.2], [-1.57, 1.4]]) {
      const dir = fromAngles(yaw!, pitch!);
      const c: Camera3D = { target: [1, -2, 0.5], distance: 7, rot: quatLook(dir), fov: 0.7 };
      const eye = cameraEye(c);
      close(eye, [1 + 7 * dir[0]!, -2 + 7 * dir[1]!, 0.5 + 7 * dir[2]!]);
      close(cameraMatrices(c, 1.5, 3).view, lookAt(eye, c.target, [0, 0, 1]));
    }
  });
  it("looking along z uses y as up: x right, y up", () => {
    const { right, up, back } = cameraAxes({ target: [0, 0, 0], distance: 1, rot: quatLook([0, 0, 1]), fov: 0.7 });
    close(right, [1, 0, 0]); close(up, [0, 1, 0]); close(back, [0, 0, 1]);
  });
  it("axis-angle rotations compose and rotate vectors", () => {
    const q = quatAxisAngle([0, 0, 1], Math.PI / 2);
    close(quatRotate(q, [1, 0, 0]), [0, 1, 0]);
    const q2 = quatMul(q, q); // 180°
    close(quatRotate(q2, [1, 0, 0]), [-1, 0, 0]);
    close(quatRotate(quatAxisAngle([1, 1, 1], 2 * Math.PI / 3), [1, 0, 0]), [0, 1, 0]);
    close(quatNormalize([0, 0, 0, 2]), [0, 0, 0, 1]);
  });
  it("a pitch orbit rolls over the pole and keeps turning", () => {
    // the viewer's orbit: pitch = rotation about the camera's right axis, applied in world space
    let rot: Quat = quatLook(fromAngles(-0.9, 0.55));
    const right0 = quatRotate(rot, [1, 0, 0]);
    const n = 126, step = (2 * Math.PI) / n;
    let crossedPole = false, upsideDown = false;
    for (let i = 0; i < n; i++) {
      const { right, up, back } = cameraAxes({ target: [0, 0, 0], distance: 1, rot, fov: 1 });
      close(right, right0, 1e-4); // the right axis is the rotation axis: it never moves
      expect(Math.abs(Math.hypot(...rot) - 1)).toBeLessThan(1e-6);
      if (back[2] > 0.999) crossedPole = true;
      if (up[2] < -0.5) upsideDown = true;
      rot = quatNormalize(quatMul(quatAxisAngle(right, -step), rot));
    }
    expect(crossedPole).toBe(true);
    expect(upsideDown).toBe(true);
    close(quatRotate(rot, [0, 0, 1]), quatRotate(quatLook(fromAngles(-0.9, 0.55)), [0, 0, 1]), 1e-3); // a full turn returns
  });
});
