/**
 * Arabic Sign Language - Skeleton Avatar
 * Three.js-based real-time skeleton with smooth interpolation
 */

import * as THREE from 'three';

export class SkeletonAvatar {
  constructor(canvas) {
    this.canvas = canvas;
    this.bones = {};
    this.targetRotations = {};
    this.currentRotations = {};
    this.animationQueue = [];
    this.isAnimating = false;
    this.currentFrame = 0;
    this.animStartTime = null;
    this.currentSign = null;
    this.onSignComplete = null;
    this.lerpSpeed = 0.12; // Controls smoothness vs speed

    this._setupScene();
    this._buildSkeleton();
    this._setupLighting();
    this._setupParticles();
    this._startRenderLoop();
  }

  /* ── Scene Setup ─────────────────────────────────── */
  _setupScene() {
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this._resize();

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0f1629);
    this.scene.fog = new THREE.FogExp2(0x0f1629, 0.04);

    this.camera = new THREE.PerspectiveCamera(50, this.canvas.clientWidth / this.canvas.clientHeight, 0.1, 100);
    this.camera.position.set(0, 1.2, 4.5);
    this.camera.lookAt(0, 0.8, 0);

    window.addEventListener('resize', () => this._resize());
  }

  _resize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    this.renderer.setSize(w, h, false);
    if (this.camera) {
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
  }

  _setupLighting() {
    const ambient = new THREE.AmbientLight(0x1a2040, 0.8);
    this.scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0x4f8ef7, 1.5);
    keyLight.position.set(3, 5, 3);
    keyLight.castShadow = true;
    this.scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x7c3aed, 0.7);
    fillLight.position.set(-3, 3, -2);
    this.scene.add(fillLight);

    const rimLight = new THREE.PointLight(0x4f8ef7, 1.2, 8);
    rimLight.position.set(0, 4, -2);
    this.scene.add(rimLight);

    // Floor glow
    const floorLight = new THREE.PointLight(0x4f8ef7, 0.5, 4);
    floorLight.position.set(0, -1, 0);
    this.scene.add(floorLight);
  }

  /* ── Skeleton Construction ───────────────────────── */
  _buildSkeleton() {
    const boneColor = 0x4f8ef7;
    const jointColor = 0x7c3aed;

    this.skeletonGroup = new THREE.Group();
    this.scene.add(this.skeletonGroup);

    // Material for bones
    this.boneMat = new THREE.MeshPhongMaterial({
      color: boneColor,
      emissive: 0x1a3566,
      shininess: 120,
    });

    // Material for joints (spheres)
    this.jointMat = new THREE.MeshPhongMaterial({
      color: jointColor,
      emissive: 0x3d1a7a,
      shininess: 200,
    });

    // Glowing material for active joints
    this.glowMat = new THREE.MeshPhongMaterial({
      color: 0xffffff,
      emissive: 0x4f8ef7,
      emissiveIntensity: 0.5,
      shininess: 300,
    });

    this._buildBody();
    this._buildArm('r', 0.6);    // right arm
    this._buildArm('l', -0.6);   // left arm
    this._buildHand('r', 0.6);
    this._buildHand('l', -0.6);
    this._buildHead();
    this._buildFloor();
  }

  _makeBone(length, radius = 0.04) {
    const geo = new THREE.CylinderGeometry(radius * 0.7, radius, length, 8);
    geo.translate(0, length / 2, 0);
    return new THREE.Mesh(geo, this.boneMat);
  }

  _makeJoint(radius = 0.07) {
    const geo = new THREE.SphereGeometry(radius, 12, 12);
    return new THREE.Mesh(geo, this.jointMat);
  }

  _buildBody() {
    // Torso
    const torso = this._makeBone(0.7, 0.06);
    torso.position.set(0, 0.4, 0);
    this.skeletonGroup.add(torso);
    this.bones['torso'] = torso;

    // Hips
    const hips = this._makeBone(0.5, 0.055);
    hips.position.set(0, 0, 0);
    this.skeletonGroup.add(hips);

    // Neck
    const neck = this._makeBone(0.18, 0.04);
    neck.position.set(0, 1.1, 0);
    this.skeletonGroup.add(neck);
    this.bones['neck'] = neck;

    // Shoulder line
    const shoulderLine = new THREE.Mesh(
      new THREE.CylinderGeometry(0.03, 0.03, 1.3, 8),
      this.boneMat
    );
    shoulderLine.rotation.z = Math.PI / 2;
    shoulderLine.position.set(0, 1.08, 0);
    this.skeletonGroup.add(shoulderLine);

    // Joint dots
    [[0, 0.4, 0], [0, 1.1, 0], [0, 1.08, 0], [-0.65, 1.08, 0], [0.65, 1.08, 0]].forEach(pos => {
      const j = this._makeJoint(0.06);
      j.position.set(...pos);
      this.skeletonGroup.add(j);
    });
  }

  _buildArm(side, xOffset) {
    const dir = side === 'r' ? 1 : -1;

    // Upper arm group (pivots at shoulder)
    const upperArmGroup = new THREE.Group();
    upperArmGroup.position.set(xOffset, 1.08, 0);
    this.skeletonGroup.add(upperArmGroup);
    this.bones[`${side}_upper_arm`] = upperArmGroup;

    const upperArm = this._makeBone(0.38, 0.045);
    upperArm.rotation.z = -dir * (Math.PI / 2) * 0.3; // slight natural angle
    upperArmGroup.add(upperArm);

    // Elbow joint
    const elbow = this._makeJoint(0.06);
    elbow.position.set(dir * 0.32, -0.1, 0);
    upperArmGroup.add(elbow);
    this.bones[`${side}_elbow_joint`] = elbow;

    // Lower arm group (pivots at elbow)
    const lowerArmGroup = new THREE.Group();
    lowerArmGroup.position.set(dir * 0.32, -0.1, 0);
    upperArmGroup.add(lowerArmGroup);
    this.bones[`${side}_lower_arm`] = lowerArmGroup;

    const lowerArm = this._makeBone(0.32, 0.035);
    lowerArm.rotation.z = -dir * (Math.PI / 2) * 0.2;
    lowerArmGroup.add(lowerArm);

    // Wrist joint
    const wrist = this._makeJoint(0.05);
    wrist.position.set(dir * 0.26, -0.08, 0);
    lowerArmGroup.add(wrist);
    this.bones[`${side}_wrist`] = wrist;
  }

  _buildHand(side, xOffset) {
    const dir = side === 'r' ? 1 : -1;
    const handGroup = new THREE.Group();
    // Position relative to wrist (inside lowerArmGroup)
    const lowerArm = this.bones[`${side}_lower_arm`];

    handGroup.position.set(dir * 0.26, -0.08, 0);
    lowerArm.add(handGroup);
    this.bones[`${side}_hand`] = handGroup;

    // Palm
    const palm = new THREE.Mesh(
      new THREE.BoxGeometry(0.1, 0.14, 0.04),
      this.boneMat
    );
    palm.position.set(dir * 0.06, 0, 0);
    handGroup.add(palm);

    // Fingers (4 + thumb)
    const fingerOffsets = [0.04, 0.013, -0.013, -0.04];
    fingerOffsets.forEach((fz, i) => {
      const finger = new THREE.Group();
      finger.position.set(dir * 0.13, 0, fz);
      handGroup.add(finger);

      // 3 phalanges per finger
      for (let p = 0; p < 3; p++) {
        const seg = this._makeBone(0.04, 0.012);
        seg.position.set(dir * p * 0.042, 0, 0);
        finger.add(seg);
      }
    });

    // Thumb
    const thumb = new THREE.Group();
    thumb.position.set(dir * 0.06, -0.07, 0.025);
    thumb.rotation.z = dir * 0.4;
    handGroup.add(thumb);
    const thumbSeg = this._makeBone(0.055, 0.014);
    thumbSeg.rotation.z = dir * 0.5;
    thumb.add(thumbSeg);
  }

  _buildHead() {
    const headGroup = new THREE.Group();
    headGroup.position.set(0, 1.28, 0);
    this.skeletonGroup.add(headGroup);
    this.bones['head'] = headGroup;

    // Skull wireframe-ish
    const skull = new THREE.Mesh(
      new THREE.SphereGeometry(0.14, 16, 16),
      new THREE.MeshPhongMaterial({
        color: 0x1a2f5c,
        emissive: 0x0d1a3a,
        wireframe: false,
        transparent: true,
        opacity: 0.85,
        shininess: 80,
      })
    );
    headGroup.add(skull);

    // Eyes
    [-0.05, 0.05].forEach(ex => {
      const eye = new THREE.Mesh(
        new THREE.SphereGeometry(0.02, 8, 8),
        new THREE.MeshPhongMaterial({ color: 0x4f8ef7, emissive: 0x2a5abf })
      );
      eye.position.set(ex, 0.02, 0.12);
      headGroup.add(eye);
    });
  }

  _buildFloor() {
    // Grid floor
    const grid = new THREE.GridHelper(6, 20, 0x1e2d4d, 0x141d35);
    grid.position.y = -0.6;
    this.scene.add(grid);

    // Glow circle under avatar
    const glowGeo = new THREE.CircleGeometry(0.6, 32);
    const glowMat = new THREE.MeshBasicMaterial({
      color: 0x4f8ef7,
      transparent: true,
      opacity: 0.08,
    });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    glow.rotation.x = -Math.PI / 2;
    glow.position.y = -0.59;
    this.scene.add(glow);
    this.floorGlow = glow;
  }

  _setupParticles() {
    // Floating dust particles
    const count = 80;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count * 3; i++) {
      pos[i] = (Math.random() - 0.5) * 6;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ color: 0x4f8ef7, size: 0.015, transparent: true, opacity: 0.4 });
    this.particles = new THREE.Points(geo, mat);
    this.scene.add(this.particles);
  }

  /* ── Animation System ────────────────────────────── */
  
  /**
   * Queue a sign for animation
   * @param {Object} signData - From arabic_signs.json
   */
  queueSign(signData) {
    this.animationQueue.push(signData);
    if (!this.isAnimating) this._processQueue();
  }

  _processQueue() {
    if (this.animationQueue.length === 0) {
      this.isAnimating = false;
      this.currentSign = null;
      if (this.onQueueEmpty) this.onQueueEmpty();
      return;
    }
    this.isAnimating = true;
    this.currentSign = this.animationQueue.shift();
    this.animStartTime = performance.now();
    if (this.onSignStart) this.onSignStart(this.currentSign.label);
  }

  _updateAnimation(now) {
    if (!this.isAnimating || !this.currentSign) return;

    const elapsed = (now - this.animStartTime) / 1000;
    const sign = this.currentSign;
    const frames = sign.frames;

    // Find surrounding keyframes
    let f0 = frames[0], f1 = frames[frames.length - 1];
    for (let i = 0; i < frames.length - 1; i++) {
      if (elapsed >= frames[i].t && elapsed <= frames[i + 1].t) {
        f0 = frames[i];
        f1 = frames[i + 1];
        break;
      }
    }

    // Interpolation factor
    const dur = f1.t - f0.t;
    const alpha = dur > 0 ? Math.min((elapsed - f0.t) / dur, 1.0) : 1.0;
    const eased = this._easeInOut(alpha);

    // Apply rotations to bones
    this._interpolateBones(f0.bones, f1.bones, eased);

    // Sign complete
    if (elapsed >= sign.duration) {
      this._returnToRest();
      if (this.onSignComplete) this.onSignComplete(sign.label);
      setTimeout(() => this._processQueue(), 180); // Brief gap between signs
    }
  }

  _interpolateBones(bones0, bones1, alpha) {
    const allKeys = new Set([...Object.keys(bones0), ...Object.keys(bones1)]);
    allKeys.forEach(key => {
      const bone = this.bones[key];
      if (!bone) return;
      const r0 = bones0[key] || [0, 0, 0];
      const r1 = bones1[key] || [0, 0, 0];

      const rx = THREE.MathUtils.lerp(r0[0], r1[0], alpha);
      const ry = THREE.MathUtils.lerp(r0[1], r1[1], alpha);
      const rz = THREE.MathUtils.lerp(r0[2], r1[2], alpha);

      // Smooth target application
      if (!this.targetRotations[key]) this.targetRotations[key] = { x: 0, y: 0, z: 0 };
      this.targetRotations[key] = {
        x: THREE.MathUtils.degToRad(rx),
        y: THREE.MathUtils.degToRad(ry),
        z: THREE.MathUtils.degToRad(rz),
      };
    });
  }

  _applyRotationsSmooth() {
    Object.entries(this.targetRotations).forEach(([key, target]) => {
      const bone = this.bones[key];
      if (!bone) return;
      bone.rotation.x = THREE.MathUtils.lerp(bone.rotation.x, target.x, this.lerpSpeed);
      bone.rotation.y = THREE.MathUtils.lerp(bone.rotation.y, target.y, this.lerpSpeed);
      bone.rotation.z = THREE.MathUtils.lerp(bone.rotation.z, target.z, this.lerpSpeed);
    });
  }

  _returnToRest() {
    this.targetRotations = {};
    Object.values(this.bones).forEach(bone => {
      this.targetRotations[Object.keys(this.bones).find(k => this.bones[k] === bone)] = { x: 0, y: 0, z: 0 };
    });
  }

  _easeInOut(t) {
    return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
  }

  /* ── Idle Animation ──────────────────────────────── */
  _applyIdleMotion(time) {
    if (this.isAnimating) return;

    const t = time * 0.001;

    // Gentle breathing
    if (this.bones['torso']) {
      this.bones['torso'].scale.y = 1 + Math.sin(t * 0.8) * 0.01;
    }

    // Subtle head movement
    if (this.bones['head']) {
      this.bones['head'].rotation.y = Math.sin(t * 0.4) * 0.04;
      this.bones['head'].rotation.z = Math.sin(t * 0.3) * 0.02;
    }

    // Arms rest gently
    if (this.bones['r_upper_arm']) {
      this.bones['r_upper_arm'].rotation.z = -0.1 + Math.sin(t * 0.5) * 0.015;
    }
    if (this.bones['l_upper_arm']) {
      this.bones['l_upper_arm'].rotation.z = 0.1 + Math.sin(t * 0.5 + 0.5) * 0.015;
    }
  }

  /* ── Render Loop ─────────────────────────────────── */
  _startRenderLoop() {
    const loop = (time) => {
      requestAnimationFrame(loop);

      // Update animations
      this._updateAnimation(time);
      this._applyRotationsSmooth();
      this._applyIdleMotion(time);

      // Rotate particles slowly
      if (this.particles) {
        this.particles.rotation.y += 0.0005;
      }

      // Pulse floor glow
      if (this.floorGlow) {
        this.floorGlow.material.opacity = 0.05 + Math.sin(time * 0.003) * 0.03;
      }

      // Slight camera sway
      this.camera.position.x = Math.sin(time * 0.0002) * 0.15;
      this.camera.lookAt(0, 0.8, 0);

      this.renderer.render(this.scene, this.camera);
    };
    requestAnimationFrame(loop);
  }

  /* ── Public API ──────────────────────────────────── */
  setLerpSpeed(speed) {
    this.lerpSpeed = speed; // 0.05 = slow/smooth, 0.25 = fast/snappy
  }

  clearQueue() {
    this.animationQueue = [];
    this.isAnimating = false;
    this._returnToRest();
  }

  get queueLength() {
    return this.animationQueue.length + (this.isAnimating ? 1 : 0);
  }
}
