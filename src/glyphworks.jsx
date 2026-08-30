import { useState, useRef, useEffect, useCallback } from "react";
import * as THREE from "three";

/* ============================================================
   GLB parsing — self-contained. No GLTFLoader dependency.
   Reads the binary glTF container, walks the node graph, and
   returns real THREE.BufferGeometry with baked world transforms.
   ============================================================ */

const TYPE_N = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };
const COMP = {
  5120: Int8Array, 5121: Uint8Array, 5122: Int16Array,
  5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array,
};

function parseGLB(buffer) {
  const dv = new DataView(buffer);
  if (buffer.byteLength < 20 || dv.getUint32(0, true) !== 0x46546c67)
    throw new Error("That file isn't a .glb. Export your model as binary glTF and try again.");
  let off = 12, json = null, bin = null;
  while (off + 8 <= buffer.byteLength) {
    const len = dv.getUint32(off, true);
    const type = dv.getUint32(off + 4, true);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, off + 8, len)));
    else if (type === 0x004e4942) bin = buffer.slice(off + 8, off + 8 + len);
    off += 8 + len + ((4 - (len % 4)) % 4);
  }
  if (!json) throw new Error("No glTF data found inside that file.");
  const req = json.extensionsRequired || [];
  if (req.some((e) => /draco/i.test(e)))
    throw new Error("This model uses Draco compression. Re-export it with compression turned off.");
  if (req.some((e) => /meshopt/i.test(e)))
    throw new Error("This model uses Meshopt compression. Re-export it without it.");
  return { json, bin };
}

function readAccessor(json, bin, index) {
  const acc = json.accessors[index];
  const n = TYPE_N[acc.type];
  const TA = COMP[acc.componentType];
  const out = new (acc.componentType === 5126 ? Float32Array : TA)(acc.count * n);
  if (acc.bufferView === undefined) return out;
  const bv = json.bufferViews[acc.bufferView];
  const base = (bv.byteOffset || 0) + (acc.byteOffset || 0);
  const packed = n * TA.BYTES_PER_ELEMENT;
  const stride = bv.byteStride || packed;
  if (stride === packed) {
    out.set(new TA(bin, base, acc.count * n));
  } else {
    for (let i = 0; i < acc.count; i++) out.set(new TA(bin, base + i * stride, n), i * n);
  }
  return out;
}

// Images live inside the binary chunk. Decode them once and hand back a
// texture per glTF image index.
async function loadTextures(json, bin) {
  if (!json.images || !json.images.length) return [];
  return Promise.all(
    json.images.map(async (img) => {
      try {
        let blob;
        if (img.bufferView !== undefined) {
          const bv = json.bufferViews[img.bufferView];
          blob = new Blob([new Uint8Array(bin, bv.byteOffset || 0, bv.byteLength)], {
            type: img.mimeType || "image/png",
          });
        } else if (img.uri && img.uri.startsWith("data:")) {
          blob = await (await fetch(img.uri)).blob();
        } else return null;
        const bitmap = await createImageBitmap(blob);
        const tex = new THREE.Texture(bitmap);
        tex.flipY = false;
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        if ("SRGBColorSpace" in THREE) tex.colorSpace = THREE.SRGBColorSpace;
        else if ("sRGBEncoding" in THREE) tex.encoding = THREE.sRGBEncoding;
        tex.needsUpdate = true;
        return tex;
      } catch {
        return null;
      }
    })
  );
}

function pick(json, textures, ref, linear) {
  if (!ref || !json.textures || !textures) return null;
  const t = json.textures[ref.index];
  const tex = t && textures[t.source];
  if (tex && linear) {
    const clone = tex.clone();
    if ("SRGBColorSpace" in THREE) clone.colorSpace = THREE.NoColorSpace;
    else clone.encoding = 3000;
    clone.needsUpdate = true;
    return clone;
  }
  return tex || null;
}

function materialInfo(json, matIndex, textures) {
  const m = (json.materials || [])[matIndex] || {};
  const pbr = m.pbrMetallicRoughness || {};
  const f = pbr.baseColorFactor;
  let color = f ? new THREE.Color(f[0], f[1], f[2]) : new THREE.Color(0xffffff);
  const map = pick(json, textures, pbr.baseColorTexture, false);
  const normalMap = pick(json, textures, m.normalTexture, true);
  const emissiveMap = pick(json, textures, m.emissiveTexture, false);
  const e = m.emissiveFactor;
  const emissive = e ? new THREE.Color(e[0], e[1], e[2]) : new THREE.Color(0, 0, 0);
  // A near-black base colour with no texture is an export artefact far more
  // often than an intent, and it renders as a silhouette with no glyphs.
  if (!map && 0.299 * color.r + 0.587 * color.g + 0.114 * color.b < 0.08)
    color = new THREE.Color(0.6, 0.6, 0.62);
  if (map) color = f ? color : new THREE.Color(0xffffff);
  return { color, map, normalMap, emissiveMap, emissive };
}

function buildModel(json, bin, textures) {
  const group = new THREE.Group();
  const nodes = json.nodes || [];
  const sceneDef = json.scenes ? json.scenes[json.scene || 0] : null;
  const roots = sceneDef && sceneDef.nodes ? sceneDef.nodes : nodes.map((_, i) => i);
  let tris = 0;

  const walk = (idx, parentMat) => {
    const node = nodes[idx];
    if (!node) return;
    const local = new THREE.Matrix4();
    if (node.matrix) local.fromArray(node.matrix);
    else
      local.compose(
        new THREE.Vector3().fromArray(node.translation || [0, 0, 0]),
        new THREE.Quaternion().fromArray(node.rotation || [0, 0, 0, 1]),
        new THREE.Vector3().fromArray(node.scale || [1, 1, 1])
      );
    const world = new THREE.Matrix4().multiplyMatrices(parentMat, local);

    if (node.mesh !== undefined && json.meshes[node.mesh]) {
      for (const prim of json.meshes[node.mesh].primitives) {
        if (prim.mode !== undefined && prim.mode !== 4) continue;
        if (!prim.attributes || prim.attributes.POSITION === undefined) continue;
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(readAccessor(json, bin, prim.attributes.POSITION), 3));
        if (prim.attributes.NORMAL !== undefined)
          geo.setAttribute("normal", new THREE.BufferAttribute(readAccessor(json, bin, prim.attributes.NORMAL), 3));
        if (prim.attributes.TEXCOORD_0 !== undefined)
          geo.setAttribute("uv", new THREE.BufferAttribute(readAccessor(json, bin, prim.attributes.TEXCOORD_0), 2));
        if (prim.indices !== undefined) {
          const idxArr = readAccessor(json, bin, prim.indices);
          geo.setIndex(new THREE.BufferAttribute(idxArr.constructor === Float32Array ? new Uint32Array(idxArr) : idxArr, 1));
        }
        if (!geo.attributes.normal) geo.computeVertexNormals();
        tris += (geo.index ? geo.index.count : geo.attributes.position.count) / 3;
        const info = materialInfo(json, prim.material, textures);
        if (!geo.attributes.uv) { info.map = null; info.normalMap = null; info.emissiveMap = null; }
        const mesh = new THREE.Mesh(
          geo,
          new THREE.MeshPhongMaterial({ color: 0xffffff, side: THREE.DoubleSide, shininess: 40 })
        );
        mesh.userData.info = info;
        mesh.applyMatrix4(world);
        group.add(mesh);
      }
    }
    (node.children || []).forEach((c) => walk(c, world));
  };

  roots.forEach((r) => walk(r, new THREE.Matrix4()));
  if (group.children.length === 0) throw new Error("No triangle meshes found in that model.");
  return { group, tris: Math.round(tris) };
}

const WHITE = new THREE.Color(0xffffff);
const BLACK = new THREE.Color(0x000000);

function applySurface(root, cfg) {
  root.traverse((o) => {
    if (!o.isMesh || !o.material) return;
    const info = o.userData.info;
    if (!o.userData.shaded) o.userData.shaded = o.material;
    const m = o.userData.shaded;
    if (info) {
      m.color.copy(cfg.albedo ? info.color : WHITE);
      m.map = cfg.albedo ? info.map : null;
      m.emissive.copy(cfg.albedo ? info.emissive : BLACK);
      m.emissiveMap = cfg.albedo ? info.emissiveMap : null;
      m.normalMap = cfg.bump ? info.normalMap : null;
    }
    if (m.specular) {
      m.specular.setScalar(cfg.spec);
      m.shininess = 6 + cfg.spec * 160;
    }
    m.needsUpdate = true;

    // Unlit twin: pure material colour, untouched by lights or highlights.
    if (!o.userData.flat) o.userData.flat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
    const f = o.userData.flat;
    f.color.copy(info ? info.color : WHITE);
    f.map = info ? info.map : null;
    f.needsUpdate = true;
    o.material = m;
  });
}

function swapMaterials(root, which) {
  root.traverse((o) => {
    if (!o.isMesh) return;
    const t = which === "flat" ? o.userData.flat : o.userData.shaded;
    if (t) o.material = t;
  });
}

function sat255(x) {
  return x < 0 ? 0 : x > 255 ? 255 : Math.round(x);
}

function fitToUnitBox(group) {
  const box = new THREE.Box3().setFromObject(group);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const max = Math.max(size.x, size.y, size.z) || 1;
  group.position.sub(center);
  const wrapper = new THREE.Group();
  wrapper.add(group);
  wrapper.scale.setScalar(2 / max);
  return wrapper;
}

/* ============================================================
   Built-in stand-ins so the page is alive before any upload
   ============================================================ */

function primitive(kind) {
  const mat = new THREE.MeshPhongMaterial({ color: 0xffffff, side: THREE.DoubleSide, shininess: 40 });
  let g;
  if (kind === "knot") g = new THREE.TorusKnotGeometry(0.72, 0.26, 220, 32);
  else if (kind === "torus") g = new THREE.TorusGeometry(0.8, 0.34, 48, 96);
  else if (kind === "ico") g = new THREE.IcosahedronGeometry(1, 1);
  else if (kind === "sphere") g = new THREE.SphereGeometry(1, 64, 48);
  else g = new THREE.BoxGeometry(1.3, 1.3, 1.3, 4, 4, 4);
  const group = new THREE.Group();
  group.add(new THREE.Mesh(g, mat));
  const tris = (g.index ? g.index.count : g.attributes.position.count) / 3;
  return { group: fitToUnitBox(group), tris: Math.round(tris) };
}

/* ============================================================
   Glyph ramps
   ============================================================ */

const RAMPS = {
  standard: " .:-=+*#%@",
  fine: " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
  blocks: " ░▒▓█",
  minimal: " .*#",
  binary: " ..01",
  arrows: " ·-+↑↗→↘↓█",
  cash: " .,:;$§%&@",
};

const EDGE_GLYPHS = ["-", "\\", "|", "/"];

/* ============================================================
   Themes
   ============================================================ */

const THEMES = {
  cobalt: { bg: "#0B0D10", tint: "#D6DDE5" },
  plotter: { bg: "#101418", tint: "#5B8CFF" },
  sulfur: { bg: "#12100A", tint: "#E8B44A" },
  paper: { bg: "#E9E7E0", tint: "#1A1A1A" },
  bone: { bg: "#141414", tint: "#F2EDE4" },
  ultra: { bg: "#0A0A12", tint: "#FF4FD8" },
};

const DEFAULTS = {
  mark: "dots",
  shape: "round",
  dot: 1.0,
  detail: 132,
  contrast: 1.2,
  brightness: 0.0,
  gamma: 1.0,
  invert: false,
  pass: "shaded",
  edges: false,
  edgeAmount: 0.22,
  color: "model",
  tint: "#D6DDE5",
  bg: "#0B0D10",
  ramp: "standard",
  custom: "",
  scale: 1.0,
  height: 0.0,
  zoom: 4.0,
  spin: 0.3,
  ambient: 0.3,
  key: 1.05,
  fill: 0.35,
  keyAngle: 45,
  albedo: true,
  bump: true,
  spec: 0.12,
  flat: true,
  sat: 1.5,
  lineHeight: 1.0,
  scanlines: false,
  glow: false,
};


/* ============================================================
   Point cloud — area-weighted surface sampling
   Vertices alone cluster wherever the modeller happened to add
   detail, so points are scattered across triangle area instead.
   ============================================================ */

const toLinear = "SRGBColorSpace" in THREE
  ? (v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4))
  : (v) => v;

function makeSampler(tex, cache) {
  if (!tex || !tex.image) return null;
  if (cache.has(tex)) return cache.get(tex);
  let fn = null;
  try {
    const img = tex.image;
    const w = Math.min(img.width || 512, 1024);
    const h = Math.min(img.height || 512, 1024);
    const cv = document.createElement("canvas");
    cv.width = w;
    cv.height = h;
    const c = cv.getContext("2d", { willReadFrequently: true });
    c.drawImage(img, 0, 0, w, h);
    const d = c.getImageData(0, 0, w, h).data;
    fn = (u, v, out) => {
      const x = ((u % 1) + 1) % 1;
      const y = ((v % 1) + 1) % 1;
      const i = (Math.min(h - 1, (y * h) | 0) * w + Math.min(w - 1, (x * w) | 0)) * 4;
      out[0] = d[i] / 255;
      out[1] = d[i + 1] / 255;
      out[2] = d[i + 2] / 255;
    };
  } catch {
    fn = null;
  }
  cache.set(tex, fn);
  return fn;
}

function samplePoints(root, cfg) {
  root.updateMatrixWorld(true);
  // Sample into the parent's space, because the Points object is added as a
  // sibling of the model group rather than a child of it.
  const inv = new THREE.Matrix4();
  if (root.parent) {
    root.parent.updateMatrixWorld(true);
    inv.copy(root.parent.matrixWorld).invert();
  }
  const cache = new Map();
  const parts = [];

  root.traverse((o) => {
    if (!o.isMesh || !o.geometry || !o.geometry.attributes.position) return;
    const g = o.geometry;
    const m = new THREE.Matrix4().multiplyMatrices(inv, o.matrixWorld);
    const nm = new THREE.Matrix3().getNormalMatrix(m);
    const src = g.attributes.position.array;
    const pos = new Float32Array(src.length);
    const v = new THREE.Vector3();
    for (let i = 0; i < src.length; i += 3) {
      v.set(src[i], src[i + 1], src[i + 2]).applyMatrix4(m);
      pos[i] = v.x; pos[i + 1] = v.y; pos[i + 2] = v.z;
    }
    let nrm = null;
    if (g.attributes.normal) {
      const ns = g.attributes.normal.array;
      nrm = new Float32Array(ns.length);
      for (let i = 0; i < ns.length; i += 3) {
        v.set(ns[i], ns[i + 1], ns[i + 2]).applyMatrix3(nm).normalize();
        nrm[i] = v.x; nrm[i + 1] = v.y; nrm[i + 2] = v.z;
      }
    }
    const info = o.userData.info;
    parts.push({
      pos,
      nrm,
      uv: g.attributes.uv ? g.attributes.uv.array : null,
      index: g.index ? g.index.array : null,
      triCount: g.index ? g.index.count / 3 : src.length / 9,
      sampler: makeSampler(info && info.map, cache),
      base: info ? info.color : WHITE,
    });
  });

  // cumulative triangle area across every part, for uniform coverage
  let tris = 0;
  for (const p of parts) tris += p.triCount;
  if (!tris) return null;
  const cum = new Float64Array(tris);
  const owner = new Int32Array(tris);
  const local = new Int32Array(tris);
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const ab = new THREE.Vector3(), ac = new THREE.Vector3();
  let t = 0, run = 0;
  parts.forEach((p, pi) => {
    for (let i = 0; i < p.triCount; i++) {
      const i0 = p.index ? p.index[i * 3] : i * 3;
      const i1 = p.index ? p.index[i * 3 + 1] : i * 3 + 1;
      const i2 = p.index ? p.index[i * 3 + 2] : i * 3 + 2;
      a.fromArray(p.pos, i0 * 3); b.fromArray(p.pos, i1 * 3); c.fromArray(p.pos, i2 * 3);
      run += ab.subVectors(b, a).cross(ac.subVectors(c, a)).length() * 0.5;
      cum[t] = run; owner[t] = pi; local[t] = i; t++;
    }
  });
  if (run <= 0) return null;

  const n = Math.max(1000, Math.floor(cfg.count));
  const position = new Float32Array(n * 3);
  const color = new Float32Array(n * 3);
  const dir = new Float32Array(n * 3);
  const box = new THREE.Box3().setFromObject(root).applyMatrix4(inv);
  const yMin = box.min.y, ySpan = Math.max(1e-6, box.max.y - box.min.y);
  const gA = new THREE.Color(cfg.gradA), gB = new THREE.Color(cfg.gradB);
  const solid = new THREE.Color(cfg.solid);
  const texel = [1, 1, 1];

  for (let k = 0; k < n; k++) {
    // pick a triangle proportional to its area
    const target = Math.random() * run;
    let lo = 0, hi = tris - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cum[mid] < target) lo = mid + 1; else hi = mid;
    }
    const p = parts[owner[lo]];
    const ti = local[lo];
    const i0 = p.index ? p.index[ti * 3] : ti * 3;
    const i1 = p.index ? p.index[ti * 3 + 1] : ti * 3 + 1;
    const i2 = p.index ? p.index[ti * 3 + 2] : ti * 3 + 2;

    let r1 = Math.sqrt(Math.random());
    const r2 = Math.random();
    const w0 = 1 - r1, w1 = r1 * (1 - r2), w2 = r1 * r2;

    const x = p.pos[i0 * 3] * w0 + p.pos[i1 * 3] * w1 + p.pos[i2 * 3] * w2;
    const y = p.pos[i0 * 3 + 1] * w0 + p.pos[i1 * 3 + 1] * w1 + p.pos[i2 * 3 + 1] * w2;
    const z = p.pos[i0 * 3 + 2] * w0 + p.pos[i1 * 3 + 2] * w1 + p.pos[i2 * 3 + 2] * w2;
    position[k * 3] = x; position[k * 3 + 1] = y; position[k * 3 + 2] = z;

    // scatter direction: along the surface normal when we have one
    if (p.nrm) {
      dir[k * 3] = p.nrm[i0 * 3] * w0 + p.nrm[i1 * 3] * w1 + p.nrm[i2 * 3] * w2;
      dir[k * 3 + 1] = p.nrm[i0 * 3 + 1] * w0 + p.nrm[i1 * 3 + 1] * w1 + p.nrm[i2 * 3 + 1] * w2;
      dir[k * 3 + 2] = p.nrm[i0 * 3 + 2] * w0 + p.nrm[i1 * 3 + 2] * w1 + p.nrm[i2 * 3 + 2] * w2;
    } else {
      dir[k * 3] = Math.random() - 0.5;
      dir[k * 3 + 1] = Math.random() - 0.5;
      dir[k * 3 + 2] = Math.random() - 0.5;
    }

    let cr = 1, cg = 1, cb = 1;
    if (cfg.colorSrc === "texture") {
      if (p.sampler && p.uv) {
        const u = p.uv[i0 * 2] * w0 + p.uv[i1 * 2] * w1 + p.uv[i2 * 2] * w2;
        const vv = p.uv[i0 * 2 + 1] * w0 + p.uv[i1 * 2 + 1] * w1 + p.uv[i2 * 2 + 1] * w2;
        p.sampler(u, vv, texel);
        cr = texel[0]; cg = texel[1]; cb = texel[2];
      } else {
        cr = p.base.r; cg = p.base.g; cb = p.base.b;
      }
      cr = toLinear(cr); cg = toLinear(cg); cb = toLinear(cb);
    } else if (cfg.colorSrc === "normal") {
      cr = dir[k * 3] * 0.5 + 0.5; cg = dir[k * 3 + 1] * 0.5 + 0.5; cb = dir[k * 3 + 2] * 0.5 + 0.5;
    } else if (cfg.colorSrc === "height") {
      const f = (y - yMin) / ySpan;
      cr = gA.r + (gB.r - gA.r) * f;
      cg = gA.g + (gB.g - gA.g) * f;
      cb = gA.b + (gB.b - gA.b) * f;
    } else {
      cr = solid.r; cg = solid.g; cb = solid.b;
    }
    color[k * 3] = cr; color[k * 3 + 1] = cg; color[k * 3 + 2] = cb;
  }

  return { position, color, dir, count: n, tris };
}

function placeVoxels(mesh, scatter) {
  const data = mesh.userData.cells;
  if (!data) return;
  const m4 = new THREE.Matrix4();
  const push = scatter * data.span;
  data.list.forEach((c, i) => {
    m4.makeTranslation(
      data.minX + (c.ix + 0.5) * data.step + c.dx * push,
      data.minY + (c.iy + 0.5) * data.step + c.dy * push,
      data.minZ + (c.iz + 0.5) * data.step + c.dz * push
    );
    mesh.setMatrixAt(i, m4);
  });
  mesh.instanceMatrix.needsUpdate = true;
  mesh.computeBoundingSphere();
}

function discTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const x = c.getContext("2d");
  // A soft shoulder means partial coverage stays partial instead of snapping
  // on and off between frames.
  const g = x.createRadialGradient(64, 64, 0, 64, 64, 62);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.62, "rgba(255,255,255,1)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  x.fillStyle = g;
  x.beginPath();
  x.arc(64, 64, 62, 0, Math.PI * 2);
  x.fill();
  const t = new THREE.CanvasTexture(c);
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  t.needsUpdate = true;
  return t;
}

const PC_DEFAULTS = {
  mres: 170,
  mdepth: 0.5,
  minv: false,
  count: 60000,
  size: 0.03,
  minPx: 1,
  atten: true,
  shape: "round",
  colorSrc: "texture",
  solid: "#7BE0FF",
  gradA: "#5B8CFF",
  gradB: "#E8B44A",
  scatter: 0,
  opacity: 1,
  additive: false,
  bg: "#07080B",
  spin: 0.25,
  zoom: 3.2,
  scale: 1,
  height: 0,
};


/* ============================================================
   Voxels — surface voxelisation by dense point sampling.
   Cheaper and far more robust than triangle/box intersection,
   and it inherits colour sampling from the point pipeline.
   ============================================================ */

function buildVoxels(root, cfg) {
  const budget = Math.min(260000, Math.max(60000, cfg.res * cfg.res * 30));
  const pts = samplePoints(root, { ...cfg, count: budget });
  if (!pts) return null;
  const P = pts.position, C = pts.color, D = pts.dir, n = pts.count;

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1;
  const step = span / cfg.res;
  const G = 512; // grid keys stay inside a safe integer
  const cells = new Map();

  for (let i = 0; i < n; i++) {
    const ix = Math.min(G - 1, Math.floor((P[i * 3] - minX) / step));
    const iy = Math.min(G - 1, Math.floor((P[i * 3 + 1] - minY) / step));
    const iz = Math.min(G - 1, Math.floor((P[i * 3 + 2] - minZ) / step));
    const key = ix + iy * G + iz * G * G;
    let c = cells.get(key);
    if (!c) { c = { ix, iy, iz, r: 0, g: 0, b: 0, dx: 0, dy: 0, dz: 0, k: 0 }; cells.set(key, c); }
    c.r += C[i * 3]; c.g += C[i * 3 + 1]; c.b += C[i * 3 + 2];
    c.dx += D[i * 3]; c.dy += D[i * 3 + 1]; c.dz += D[i * 3 + 2];
    c.k++;
  }

  const list = [...cells.values()];
  for (const c of list) {
    const len = Math.hypot(c.dx, c.dy, c.dz) || 1;
    c.dx /= len; c.dy /= len; c.dz /= len;
  }
  return { list, step, minX, minY, minZ, span };
}

/* ============================================================
   Wireframe — feature edges, with the solid drawn in the
   background colour so edges behind the form are occluded.
   ============================================================ */

function localMatrix(root, o) {
  const inv = new THREE.Matrix4();
  if (root.parent) {
    root.parent.updateMatrixWorld(true);
    inv.copy(root.parent.matrixWorld).invert();
  }
  return new THREE.Matrix4().multiplyMatrices(inv, o.matrixWorld);
}

// Only the LineSegments geometries belong to us. The occluder meshes share the
// model's own BufferGeometry — disposing those destroys the uploaded model.
function disposeWire(group) {
  if (!group) return;
  group.traverse((o) => {
    if (o.isLineSegments && o.geometry) o.geometry.dispose();
  });
  if (group.userData.lineMat) group.userData.lineMat.dispose();
  if (group.userData.solidMat) group.userData.solidMat.dispose();
}

function buildWire(root, cfg) {
  root.updateMatrixWorld(true);
  const group = new THREE.Group();
  const lineMat = new THREE.LineBasicMaterial({
    color: new THREE.Color(cfg.color),
    transparent: true,
    opacity: cfg.opacity,
    blending: cfg.additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    depthWrite: false,
    fog: true,
  });
  const solidMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(cfg.bg),
    colorWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
    fog: true,
  });
  let segs = 0;

  root.traverse((o) => {
    if (!o.isMesh || !o.geometry) return;
    const m = localMatrix(root, o);
    const geo =
      cfg.mode === "edges"
        ? new THREE.EdgesGeometry(o.geometry, cfg.angle)
        : new THREE.WireframeGeometry(o.geometry);
    segs += geo.attributes.position.count / 2;
    if (cfg.hideBack) {
      const solid = new THREE.Mesh(o.geometry, solidMat);
      solid.applyMatrix4(m);
      solid.renderOrder = 0;
      group.add(solid);
    }
    const lines = new THREE.LineSegments(geo, lineMat);
    lines.applyMatrix4(m);
    lines.renderOrder = 1;
    group.add(lines);
  });

  group.userData.segs = Math.round(segs);
  group.userData.lineMat = lineMat;
  group.userData.solidMat = solidMat;
  return group;
}

const VX_DEFAULTS = {
  res: 44,
  scatter: 0,
  gap: 0.1,
  colorSrc: "texture",
  solid: "#E8B44A",
  gradA: "#5B8CFF",
  gradB: "#E8B44A",
  bg: "#0B0D10",
  lit: true,
  ambient: 0.35,
  key: 1.1,
  spin: 0.25,
  zoom: 3.2,
  scale: 1,
  height: 0,
};

const WF_DEFAULTS = {
  mode: "edges",
  angle: 20,
  color: "#5B8CFF",
  bg: "#07080B",
  opacity: 0.9,
  hideBack: true,
  fog: 0.45,
  additive: false,
  spin: 0.25,
  zoom: 3.2,
  scale: 1,
  height: 0,
};


/* ============================================================
   Dithering — quantise a low-res render to a fixed palette and
   push the quantisation error into neighbouring pixels.
   ============================================================ */

function bayerMatrix(n) {
  let m = [[0]];
  while (m.length < n) {
    const size = m.length;
    const out = [];
    for (let y = 0; y < size * 2; y++) out.push(new Array(size * 2));
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const v = m[y][x] * 4;
        out[y][x] = v;
        out[y][x + size] = v + 2;
        out[y + size][x] = v + 3;
        out[y + size][x + size] = v + 1;
      }
    m = out;
  }
  return m;
}
const BAYER = { 2: bayerMatrix(2), 4: bayerMatrix(4), 8: bayerMatrix(8) };

const PALETTES = {
  mono: null, // built from the user's ink and field colours
  gameboy: ["#0F380F", "#306230", "#8BAC0F", "#9BBC0F"],
  cga: ["#000000", "#55FFFF", "#FF55FF", "#FFFFFF"],
  zx: ["#000000", "#0000D7", "#D70000", "#D700D7", "#00D700", "#00D7D7", "#D7D700", "#FFFFFF"],
  teletext: ["#000000", "#FF0000", "#00FF00", "#FFFF00", "#0000FF", "#FF00FF", "#00FFFF", "#FFFFFF"],
  posterise: null, // per-channel levels instead of a fixed list
};

function hexList(list) {
  return list.map((h) => {
    const n = parseInt(h.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  });
}

function nearestIn(pal, r, g, b) {
  let best = pal[0], bd = Infinity;
  for (let i = 0; i < pal.length; i++) {
    const p = pal[i];
    const dr = r - p[0], dg = g - p[1], db = b - p[2];
    const d = dr * dr * 0.3 + dg * dg * 0.59 + db * db * 0.11;
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}

const DT_DEFAULTS = {
  res: 150,
  algo: "fs",
  bayer: 4,
  spread: 1,
  palette: "mono",
  levels: 3,
  contrast: 1.2,
  brightness: 0,
  gamma: 1,
  invert: false,
  ink: "#EDE7DA",
  bg: "#0E0E12",
  ambient: 0.3,
  key: 1.15,
  spin: 0.3,
  zoom: 3.4,
  scale: 1,
  height: 0,
};


/* ============================================================
   Timeline — keyframes, interpolation, and frame-accurate export
   ============================================================ */

// Only parameters that are safe to change every frame. Anything that
// forces a resample or a geometry rebuild is deliberately excluded.
const FIELDS = { scene: null, black: "#000000", white: "#FFFFFF", green: "#00B140" };

const ANIM = {
  ascii:  { num: ["detail","contrast","brightness","gamma","lineHeight","dot","spec","sat","edgeAmount","zoom","scale","height"], col: ["tint","bg"] },
  dither: { num: ["res","contrast","brightness","gamma","spread","ambient","key","zoom","scale","height"], col: ["ink","bg"] },
  points: { num: ["size","opacity","scatter","zoom","scale","height"], col: ["bg"] },
  voxel:  { num: ["scatter","ambient","key","zoom","scale","height"], col: ["bg"] },
  wire:   { num: ["opacity","fog","zoom","scale","height"], col: ["color","bg"] },
};

const EASE = {
  linear: (t) => t,
  smooth: (t) => t * t * (3 - 2 * t),
  in: (t) => t * t,
  out: (t) => 1 - (1 - t) * (1 - t),
};

function hexToArr(h) {
  const n = parseInt(String(h).replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function arrToHex(a) {
  return "#" + a.map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0")).join("");
}
function lerpHex(a, b, f) {
  const x = hexToArr(a), y = hexToArr(b);
  return arrToHex([x[0] + (y[0] - x[0]) * f, x[1] + (y[1] - x[1]) * f, x[2] + (y[2] - x[2]) * f]);
}

/* ---- store-only ZIP, so PNG sequences come out as one file ---- */
const CRC_T = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(u8) {
  let c = 0xffffffff;
  for (let i = 0; i < u8.length; i++) c = CRC_T[(c ^ u8[i]) & 255] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function zipStore(files) {
  const enc = new TextEncoder();
  const chunks = [], central = [];
  let offset = 0;
  const mk = (n) => { const b = new Uint8Array(n); return [b, new DataView(b.buffer)]; };
  for (const f of files) {
    const name = enc.encode(f.name), crc = crc32(f.data), len = f.data.length;
    const [lh, v] = mk(30);
    v.setUint32(0, 0x04034b50, true); v.setUint16(4, 20, true);
    v.setUint32(14, crc, true); v.setUint32(18, len, true); v.setUint32(22, len, true);
    v.setUint16(26, name.length, true);
    chunks.push(lh, name, f.data);
    const [cd, cv] = mk(46);
    cv.setUint32(0, 0x02014b50, true); cv.setUint16(4, 20, true); cv.setUint16(6, 20, true);
    cv.setUint32(16, crc, true); cv.setUint32(20, len, true); cv.setUint32(24, len, true);
    cv.setUint16(28, name.length, true); cv.setUint32(42, offset, true);
    central.push(cd, name);
    offset += 30 + name.length + len;
  }
  let cSize = 0;
  for (const c of central) cSize += c.length;
  const [eo, ev] = mk(22);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, files.length, true); ev.setUint16(10, files.length, true);
  ev.setUint32(12, cSize, true); ev.setUint32(16, offset, true);
  return new Blob([...chunks, ...central, eo], { type: "application/zip" });
}


/* ============================================================
   Media sources — image and webcam
   ASCII and Dither never knew they were looking at 3D. Both read a
   flat cols×rows RGBA buffer, so a photo or a camera frame only has to
   land in that same buffer, in the same bottom-up row order WebGL uses,
   with alpha carrying the emptiness mask. Every downstream control —
   ramps, palettes, dithering, dot radius — then works untouched.
   ============================================================ */

function fillFromSource(E, cols, rows, cellW, cellH, cfg) {
  const src = E.src;
  const el = src && src.el;
  if (!el) return false;
  const sw = el.videoWidth || el.naturalWidth || el.width || 0;
  const sh = el.videoHeight || el.naturalHeight || el.height || 0;
  if (!sw || !sh) return false;

  if (!E.scv || E.scv.width !== cols || E.scv.height !== rows) {
    E.scv = document.createElement("canvas");
    E.scv.width = cols;
    E.scv.height = rows;
    E.sctx = E.scv.getContext("2d", { willReadFrequently: true });
  }
  const ctx = E.sctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, cols, rows);

  // Cells aren't square in glyph mode, so fit against the aspect the grid
  // actually displays at, not against cols/rows.
  const ga = (sw / sh) * (cellH / cellW);
  const grid = cols / rows;
  let gw, gh;
  if (cfg.fit === "cover" ? ga > grid : ga < grid) { gh = rows; gw = rows * ga; }
  else { gw = cols; gh = cols / ga; }
  const dx = (cols - gw) / 2, dy = (rows - gh) / 2;

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  if (cfg.mirror) { ctx.translate(cols, 0); ctx.scale(-1, 1); }
  try { ctx.drawImage(el, dx, dy, gw, gh); } catch { return false; }

  const d = ctx.getImageData(0, 0, cols, rows).data;
  const buf = E.buf;
  const cut = cfg.cut * 255;
  for (let r = 0; r < rows; r++) {
    const dst = (rows - 1 - r) * cols * 4; // written flipped to match readRenderTargetPixels
    const row = r * cols * 4;
    for (let c = 0; c < cols; c++) {
      const i = row + c * 4, o = dst + c * 4;
      const R = d[i], G = d[i + 1], B = d[i + 2];
      buf[o] = R; buf[o + 1] = G; buf[o + 2] = B;
      // A camera frame is opaque everywhere, so emptiness has to come from
      // luminance instead — this is what carves a subject out of a dark room.
      buf[o + 3] = d[i + 3] > 8 && 0.299 * R + 0.587 * G + 0.114 * B >= cut ? 255 : 0;
    }
  }
  return true;
}


/* ============================================================
   Media point cloud — one point per sampled pixel, brightness as depth.
   Points are compacted into the front of a fixed buffer and exposed with
   setDrawRange, so cut-out pixels cost nothing and nothing reallocates
   between frames.
   ============================================================ */

const MEDIA_CAP = 320 * 320;

function ensureMediaPoints(E) {
  if (E.mpts) return E.mpts;
  const pos = new Float32Array(MEDIA_CAP * 3);
  const col = new Float32Array(MEDIA_CAP * 3);
  const jit = new Float32Array(MEDIA_CAP * 3);
  for (let i = 0; i < jit.length; i++) jit[i] = Math.random() - 0.5;
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.setDrawRange(0, 0);
  const mat = new THREE.PointsMaterial({
    size: 0.02,
    sizeAttenuation: true,
    vertexColors: true,
    transparent: true,
    map: E.disc,
    alphaTest: 0.06,
    depthWrite: true,
  });
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uMinPx = { value: 2.5 };
    sh.vertexShader = "uniform float uMinPx;\n" + sh.vertexShader;
    sh.vertexShader = sh.vertexShader.replace(
      "#include <fog_vertex>",
      "gl_PointSize = max( gl_PointSize, uMinPx );\n\t#include <fog_vertex>"
    );
    mat.userData.sh = sh;
  };
  const pts = new THREE.Points(g, mat);
  pts.frustumCulled = false; // the buffer is rewritten every frame
  E.mpts = pts;
  E.mjit = jit;
  E.pivot.add(pts);
  return pts;
}

function updateMediaPoints(E, pcfg, scfg) {
  const el = E.src && E.src.el;
  if (!el) return 0;
  const sw = el.videoWidth || el.naturalWidth || 0;
  const sh = el.videoHeight || el.naturalHeight || 0;
  if (!sw || !sh) return 0;

  let gw = Math.max(32, Math.min(320, Math.round(pcfg.mres)));
  let gh = Math.max(24, Math.round((gw * sh) / sw));
  while (gw * gh > MEDIA_CAP) { gw = Math.floor(gw * 0.92); gh = Math.round((gw * sh) / sw); }

  if (!E.mcv || E.mcv.width !== gw || E.mcv.height !== gh) {
    E.mcv = document.createElement("canvas");
    E.mcv.width = gw;
    E.mcv.height = gh;
    E.mctx = E.mcv.getContext("2d", { willReadFrequently: true });
  }
  const ctx = E.mctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, gw, gh);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  if (scfg.mirror) { ctx.translate(gw, 0); ctx.scale(-1, 1); }
  try { ctx.drawImage(el, 0, 0, gw, gh); } catch { return 0; }

  const d = ctx.getImageData(0, 0, gw, gh).data;
  const pts = ensureMediaPoints(E);
  const pos = pts.geometry.attributes.position.array;
  const col = pts.geometry.attributes.color.array;
  const jit = E.mjit;
  const cut = scfg.cut * 255;
  const depth = pcfg.mdepth;
  const inv = pcfg.minv;
  const jitter = pcfg.scatter;
  const halfH = gh / gw;
  let k = 0;

  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const i = (y * gw + x) * 4;
      const R = d[i], G = d[i + 1], B = d[i + 2];
      if (d[i + 3] < 8) continue;
      const l = 0.299 * R + 0.587 * G + 0.114 * B;
      if (l < cut) continue;
      const nl = l / 255;
      const o = k * 3;
      pos[o] = (x / (gw - 1) - 0.5) * 2 + jit[o] * jitter;
      pos[o + 1] = (0.5 - y / (gh - 1)) * 2 * halfH + jit[o + 1] * jitter;
      pos[o + 2] = ((inv ? 1 - nl : nl) - 0.5) * depth * 2 + jit[o + 2] * jitter;
      col[o] = toLinear(R / 255);
      col[o + 1] = toLinear(G / 255);
      col[o + 2] = toLinear(B / 255);
      k++;
    }
  }
  pts.geometry.setDrawRange(0, k);
  pts.geometry.attributes.position.needsUpdate = true;
  pts.geometry.attributes.color.needsUpdate = true;
  E.mgrid = gw + "×" + gh;
  return k;
}


/* ============================================================
   Media studio — photo and video, treated as a picture rather
   than as a render source. Backdrop, marks, grade and post all
   composite on one 2D canvas.
   ============================================================ */

// Braille packs 2x4 sub-samples into a single glyph, so it carries eight
// times the detail of a character cell at the same font size.
const BRAILLE = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];
const HALFBLOCK = [" ", "\u2580", "\u2584", "\u2588"]; // none, upper, lower, full

const GRADES = {
  none: null,
  noir: (r, g, b) => { const l = 0.299 * r + 0.587 * g + 0.114 * b; return [l, l, l]; },
  sepia: (r, g, b) => [0.393 * r + 0.769 * g + 0.189 * b, 0.349 * r + 0.686 * g + 0.168 * b, 0.272 * r + 0.534 * g + 0.131 * b],
  warm: (r, g, b) => [r * 1.14 + 12, g * 1.02, b * 0.86],
  cool: (r, g, b) => [r * 0.86, g, b * 1.16 + 12],
  vintage: (r, g, b) => { const l = 0.299 * r + 0.587 * g + 0.114 * b; return [l * 0.55 + r * 0.5 + 20, l * 0.5 + g * 0.46 + 10, l * 0.46 + b * 0.36]; },
  cyber: (r, g, b) => { const l = 0.299 * r + 0.587 * g + 0.114 * b; return [l * 0.45 + r * 0.4 + 26, l * 0.8 + g * 0.28, l * 0.9 + b * 0.5 + 42]; },
};

const MIXED_BANDS = [" .:-=", "\u2591\u2592\u2593", "+*#%", "\u2588\u25A0\u25CF", "@$&"];

const STYLES = [
  ["chars", "Chars"], ["dither", "Dither"], ["blocks", "Block"], ["dots", "Dots"],
  ["mixed", "Mixed"], ["pixel", "Pixel"], ["mosaic", "Mosaic"], ["lego", "LEGO"],
  ["cross", "Cross"], ["diamond", "Diamond"], ["lines", "Lines"], ["diagonal", "Diagonal"],
  ["braille", "Braille"], ["voxel", "Voxel"], ["disco", "Disco"],
];
const GLYPH_STYLES = new Set(["chars", "blocks", "braille", "mixed"]);

const RECIPES = {
  matrix:   { style: "chars",  ramp: "fine",     tint: "#3DFF7A", bg: "#03110A", color: "mono",   backdrop: "none", grade: "none",    vignette: 0.45, scan: 0.25, cBloom: 0.5,  grain: 0.05, cChroma: 0,   crt: 0.25, glitch: 0,    halftone: 0,   dust: 0,    aberr: 0,    bloom: 0,    pixel: 0 },
  noir:     { style: "dither", ramp: "standard", tint: "#F2F0EA", bg: "#0A0A0C", color: "mono",   backdrop: "none", grade: "noir",    vignette: 0.55, scan: 0,    cBloom: 0,    grain: 0.18, cChroma: 0,   crt: 0,    glitch: 0,    halftone: 0,   dust: 0.25, aberr: 0,    bloom: 0,    pixel: 0 },
  vaporwave:{ style: "dots",   ramp: "standard", tint: "#FF6BD6", bg: "#160B2A", color: "source", backdrop: "blur", grade: "cyber",   vignette: 0.3,  scan: 0.18, cBloom: 0.55, grain: 0,    cChroma: 0.4, crt: 0,    glitch: 0.1,  halftone: 0,   dust: 0,    aberr: 0.25, bloom: 0.2,  pixel: 0 },
  terminal: { style: "chars",  ramp: "standard", tint: "#E8B44A", bg: "#0B0906", color: "mono",   backdrop: "none", grade: "none",    vignette: 0.4,  scan: 0.35, cBloom: 0.25, grain: 0.08, cChroma: 0,   crt: 0.45, glitch: 0,    halftone: 0,   dust: 0,    aberr: 0,    bloom: 0,    pixel: 0 },
  print:    { style: "braille",ramp: "standard", tint: "#141414", bg: "#E9E7E0", color: "mono",   backdrop: "none", grade: "noir",    vignette: 0,    scan: 0,    cBloom: 0,    grain: 0.06, cChroma: 0,   crt: 0,    glitch: 0,    halftone: 0.5, dust: 0,    aberr: 0,    bloom: 0,    pixel: 0 },
  polaroid: { style: "lego",   ramp: "standard", tint: "#FFFFFF", bg: "#120F0C", color: "source", backdrop: "orig", grade: "vintage", vignette: 0.5,  scan: 0,    cBloom: 0.15, grain: 0.22, cChroma: 0,   crt: 0,    glitch: 0,    halftone: 0,   dust: 0.3,  aberr: 0,    bloom: 0,    pixel: 0 },
  arcade:   { style: "pixel",  ramp: "standard", tint: "#FFFFFF", bg: "#05060A", color: "source", backdrop: "none", grade: "none",    vignette: 0.35, scan: 0.3,  cBloom: 0.3,  grain: 0.05, cChroma: 0,   crt: 0.5,  glitch: 0,    halftone: 0,   dust: 0,    aberr: 0.3,  bloom: 0.15, pixel: 0 },
  broken:   { style: "mixed",  ramp: "fine",     tint: "#9BE7FF", bg: "#06080E", color: "source", backdrop: "blur", grade: "cool",    vignette: 0.4,  scan: 0.12, cBloom: 0.35, grain: 0.1,  cChroma: 0.5, crt: 0.15, glitch: 0.45, halftone: 0,   dust: 0.15, aberr: 0.2,  bloom: 0,    pixel: 0 },
};

const MD_DEFAULTS = {
  style: "chars", detail: 150, ramp: "standard", custom: "", dot: 1, shape: "round", depth: 0.6,
  cut: 0.04, contrast: 1.15, brightness: 0, gamma: 1, invert: false,
  color: "source", tint: "#EDE7DA", bg: "#08080B", sat: 1, grade: "none",
  backdrop: "blur", blur: 10, backOpacity: 0.35,
  cBloom: 0, cChroma: 0,
  bloom: 0, aberr: 0, crt: 0, glitch: 0, pixel: 0, halftone: 0, dust: 0,
  vignette: 0.32, scan: 0, grain: 0,
  fit: "cover", mirror: false, zoom: 1, panX: 0, panY: 0,
};

function coverRect(sw, sh, W, H, mode) {
  const k = mode === "contain" ? Math.min(W / sw, H / sh) : Math.max(W / sw, H / sh);
  const w = sw * k, h = sh * k;
  return [(W - w) / 2, (H - h) / 2, w, h];
}

function scratch(E, key, w, h) {
  const c = E[key] || (E[key] = document.createElement("canvas"));
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return c;
}


function drawMarks(m, cfg, G) {
  const { cols, rows, cellW, cellH, yPad, lum, hit, flat, ramp, at } = G;
  const last = ramp.length - 1;
  // Everything is drawn in a single flat colour. Per-cell colour is applied to
  // the finished layer in one composite, so cost no longer scales with cells.
  const ink = () => flat;
  const cx = (c) => c * cellW + cellW / 2;
  const cy = (r) => yPad + r * cellH + cellH / 2;
  let text = "";

  const glyphRun = (pick) => {
    for (let r = 0; r < rows; r++) {
      let line = "";
      for (let c = 0; c < cols; c++) line += pick(c, r);
      text += line + "\n";
      m.fillStyle = flat;
      m.fillText(line, 0, yPad + r * cellH);
    }
  };

  if (cfg.style === "chars") {
    glyphRun((c, r) => { const i = at(c, r); return hit[i] ? ramp[Math.round(lum[i] * last)] || " " : " "; });
  } else if (cfg.style === "mixed") {
    glyphRun((c, r) => {
      const i = at(c, r);
      if (!hit[i]) return " ";
      const v = lum[i];
      const b = MIXED_BANDS[Math.min(MIXED_BANDS.length - 1, Math.floor(v * MIXED_BANDS.length))];
      const local = (v * MIXED_BANDS.length) % 1;
      return b[Math.min(b.length - 1, Math.floor(local * b.length))];
    });
  } else if (cfg.style === "blocks") {
    glyphRun((c, r) => {
      const t = at(c, r * 2), b = at(c, r * 2 + 1);
      return HALFBLOCK[(hit[t] && lum[t] > 0.5 ? 1 : 0) + (hit[b] && lum[b] > 0.5 ? 2 : 0)];
    });
  } else if (cfg.style === "braille") {
    glyphRun((c, r) => {
      let bits = 0;
      for (let dx = 0; dx < 2; dx++)
        for (let dy = 0; dy < 4; dy++) {
          const i = at(c * 2 + dx, r * 4 + dy);
          if (hit[i] && lum[i] > 0.5) bits |= BRAILLE[dx][dy];
        }
      return bits ? String.fromCharCode(0x2800 + bits) : " ";
    });
  } else if (cfg.style === "dots") {
    const maxR = Math.min(cellW, cellH) * 0.5 * cfg.dot;
    const ring = cfg.shape === "ring", square = cfg.shape === "square";
    const plot = (x, y, rad) => {
      if (square) m.rect(x - rad, y - rad, rad * 2, rad * 2);
      else { m.moveTo(x + rad, y); m.arc(x, y, rad, 0, Math.PI * 2); }
    };
    m.beginPath();
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = at(c, r); if (!hit[i]) continue;
      const rad = maxR * Math.sqrt(lum[i]); if (rad < 0.12) continue;
      plot(cx(c), cy(r), rad);
    }
    if (ring) { m.strokeStyle = flat; m.lineWidth = Math.max(0.6, maxR * 0.28); m.stroke(); }
    else { m.fillStyle = flat; m.fill(); }
  } else if (cfg.style === "pixel") {
    const t = hexToRgb(cfg.tint);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = at(c, r); if (!hit[i]) continue;
      m.fillStyle = `rgba(${t.r},${t.g},${t.b},${Math.round(lum[i] * 5) / 5})`;
      m.fillRect(Math.floor(c * cellW), Math.floor(yPad + r * cellH), Math.ceil(cellW), Math.ceil(cellH));
    }
  } else if (cfg.style === "mosaic") {
    const gap = Math.max(1, cellW * 0.12);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = at(c, r); if (!hit[i]) continue;
      const x = c * cellW + gap / 2, y = yPad + r * cellH + gap / 2;
      const w = cellW - gap, h = cellH - gap, rr = Math.min(w, h) * 0.18;
      m.fillStyle = ink();
      m.beginPath();
      m.moveTo(x + rr, y);
      m.arcTo(x + w, y, x + w, y + h, rr); m.arcTo(x + w, y + h, x, y + h, rr);
      m.arcTo(x, y + h, x, y, rr); m.arcTo(x, y, x + w, y, rr);
      m.closePath(); m.fill();
      m.fillStyle = "rgba(255,255,255,0.10)";
      m.fillRect(x, y, w, Math.max(1, h * 0.18));
    }
  } else if (cfg.style === "lego") {
    const gap = Math.max(1, cellW * 0.06);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = at(c, r); if (!hit[i]) continue;
      const x = c * cellW + gap / 2, y = yPad + r * cellH + gap / 2;
      const w = cellW - gap, h = cellH - gap;
      m.fillStyle = ink(); m.fillRect(x, y, w, h);
      m.fillStyle = "rgba(0,0,0,0.30)"; m.fillRect(x, y + h * 0.82, w, h * 0.18);
      const sr = Math.min(w, h) * 0.26;
      m.beginPath(); m.arc(x + w / 2, y + h / 2, sr, 0, Math.PI * 2);
      m.fillStyle = ink(); m.fill();
      m.strokeStyle = "rgba(255,255,255,0.35)";
      m.lineWidth = Math.max(0.6, sr * 0.22);
      m.beginPath(); m.arc(x + w / 2, y + h / 2, sr, Math.PI * 1.05, Math.PI * 1.95); m.stroke();
    }
  } else if (cfg.style === "cross") {
    m.fillStyle = ink();
    m.beginPath();
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = at(c, r); if (!hit[i]) continue;
      const len = Math.min(cellW, cellH) * lum[i], th = Math.max(0.7, len * 0.26);
      m.rect(cx(c) - len / 2, cy(r) - th / 2, len, th);
      m.rect(cx(c) - th / 2, cy(r) - len / 2, th, len);
    }
    m.fill();
  } else if (cfg.style === "diamond") {
    m.fillStyle = ink();
    m.beginPath();
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = at(c, r); if (!hit[i]) continue;
      const s2 = Math.min(cellW, cellH) * 0.5 * Math.sqrt(lum[i]);
      if (s2 < 0.2) continue;
      m.moveTo(cx(c), cy(r) - s2); m.lineTo(cx(c) + s2, cy(r));
      m.lineTo(cx(c), cy(r) + s2); m.lineTo(cx(c) - s2, cy(r));
      m.closePath();
    }
    m.fill();
  } else if (cfg.style === "lines") {
    m.fillStyle = ink();
    m.beginPath();
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = at(c, r); if (!hit[i]) continue;
      const w = cellW * lum[i]; if (w < 0.3) continue;
      m.rect(cx(c) - w / 2, yPad + r * cellH, w, cellH);
    }
    m.fill();
  } else if (cfg.style === "diagonal") {
    m.lineCap = "round";
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = at(c, r); if (!hit[i]) continue;
      const s2 = Math.min(cellW, cellH) * 0.5;
      m.strokeStyle = ink();
      m.lineWidth = Math.max(0.5, s2 * lum[i] * 0.9);
      m.beginPath();
      m.moveTo(cx(c) - s2, cy(r) + s2); m.lineTo(cx(c) + s2, cy(r) - s2);
      m.stroke();
    }
  } else if (cfg.style === "voxel") {
    const s2 = cellW * 0.5;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = at(c, r); if (!hit[i]) continue;
      const h = lum[i] * cellH * 2.2 * cfg.depth;
      const x = cx(c), y = cy(r) + cellH * 0.5, ty = y - h;
      const base = hexToRgb(flat);
      const sh = (k) => `rgb(${Math.min(255, base.r * k) | 0},${Math.min(255, base.g * k) | 0},${Math.min(255, base.b * k) | 0})`;
      m.fillStyle = sh(1.3);
      m.beginPath();
      m.moveTo(x, ty - s2 * 0.5); m.lineTo(x + s2, ty); m.lineTo(x, ty + s2 * 0.5); m.lineTo(x - s2, ty);
      m.closePath(); m.fill();
      m.fillStyle = sh(0.58);
      m.beginPath();
      m.moveTo(x - s2, ty); m.lineTo(x, ty + s2 * 0.5); m.lineTo(x, y + s2 * 0.5); m.lineTo(x - s2, y);
      m.closePath(); m.fill();
      m.fillStyle = sh(0.86);
      m.beginPath();
      m.moveTo(x + s2, ty); m.lineTo(x, ty + s2 * 0.5); m.lineTo(x, y + s2 * 0.5); m.lineTo(x + s2, y);
      m.closePath(); m.fill();
    }
  } else if (cfg.style === "disco") {
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = at(c, r); if (!hit[i]) continue;
      const v = lum[i], rad = Math.min(cellW, cellH) * 0.5 * Math.sqrt(v);
      if (rad < 0.3) continue;
      const g = m.createRadialGradient(cx(c) - rad * 0.35, cy(r) - rad * 0.35, rad * 0.05, cx(c), cy(r), rad);
      g.addColorStop(0, "rgba(255,255,255,0.95)");
      g.addColorStop(0.45, ink());
      g.addColorStop(1, "rgba(0,0,0,0.55)");
      m.fillStyle = g;
      m.beginPath(); m.arc(cx(c), cy(r), rad, 0, Math.PI * 2); m.fill();
      if (v > 0.82) {
        m.strokeStyle = "rgba(255,255,255,0.85)";
        m.lineWidth = Math.max(0.5, rad * 0.16);
        m.beginPath();
        m.moveTo(cx(c) - rad * 1.7, cy(r)); m.lineTo(cx(c) + rad * 1.7, cy(r));
        m.moveTo(cx(c), cy(r) - rad * 1.7); m.lineTo(cx(c), cy(r) + rad * 1.7);
        m.stroke();
      }
    }
  } else {
    const work = Float32Array.from(lum);
    for (let r = 0; r < rows; r++) {
      const ltr = r % 2 === 0;
      for (let j = 0; j < cols; j++) {
        const c = ltr ? j : cols - 1 - j;
        const i = at(c, r); if (!hit[i]) continue;
        const old = work[i], q = old > 0.5 ? 1 : 0;
        work[i] = q;
        const err = old - q, d = ltr ? 1 : -1;
        const push = (cc, rr, f) => {
          if (cc < 0 || cc >= cols || rr >= rows) return;
          const k = at(cc, rr); if (hit[k]) work[k] += err * f;
        };
        push(c + d, r, 7 / 16); push(c - d, r + 1, 3 / 16);
        push(c, r + 1, 5 / 16); push(c + d, r + 1, 1 / 16);
      }
    }
    m.fillStyle = ink();
    m.beginPath();
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = at(c, r);
      if (!hit[i] || work[i] < 0.5) continue;
      m.rect(Math.floor(c * cellW), Math.floor(yPad + r * cellH), Math.ceil(cellW), Math.ceil(cellH));
    }
    m.fill();
  }
  return text;
}


/* ============================================================
   Collage — layout and time, rather than marks. Every panel is
   treated independently, so one frame can carry nine different
   looks, or nine moments can share one sheet.
   ============================================================ */

const LOOKS = [
  { name: "ascii",    style: "chars",   ramp: "standard", grade: "none",    tint: "#EDE7DA", bg: "#0A0A0E", color: "mono" },
  { name: "halftone", style: "dots",    ramp: "standard", grade: "none",    tint: "#14140F", bg: "#F2EFE8", color: "mono" },
  { name: "noir",     style: "dither",  ramp: "standard", grade: "noir",    tint: "#F2F0EA", bg: "#0A0A0C", color: "mono" },
  { name: "pixel",    style: "pixel",   ramp: "standard", grade: "none",    tint: "#FFFFFF", bg: "#05060A", color: "source" },
  { name: "braille",  style: "braille", ramp: "standard", grade: "noir",    tint: "#141414", bg: "#E9E7E0", color: "mono" },
  { name: "lego",     style: "lego",    ramp: "standard", grade: "vintage", tint: "#FFFFFF", bg: "#120F0C", color: "source" },
  { name: "cyber",    style: "mixed",   ramp: "fine",     grade: "cyber",   tint: "#9BE7FF", bg: "#06080E", color: "source" },
  { name: "disco",    style: "disco",   ramp: "standard", grade: "warm",    tint: "#FFD36B", bg: "#140A06", color: "source" },
  { name: "photo",    style: "photo",   ramp: "standard", grade: "none",    tint: "#FFFFFF", bg: "#000000", color: "source" },
  { name: "sepia",    style: "photo",   ramp: "standard", grade: "sepia",   tint: "#FFFFFF", bg: "#000000", color: "source" },
];

const DUOS = [
  ["#FF2E63", "#FFF3B0"], ["#00E5FF", "#0E1240"], ["#FFD400", "#3B0A45"],
  ["#00FF9C", "#0B1E13"], ["#FF6B00", "#1A0A2E"], ["#F2F2F2", "#101014"],
  ["#B14CFF", "#FFE9A8"], ["#FF4D4D", "#06232E"],
  ["#3DFF7A", "#02120A"], ["#7FD4FF", "#061423"], ["#FF71CE", "#1B0B2E"],
  ["#E8B44A", "#0B0906"], ["#FFFFFF", "#0A0A0A"], ["#FF9AD5", "#2B1E3F"],
];

// Each preset is a complete sheet: layout, framing, palette, trim and post.
const CL_RECIPES = {
  cyberpunk: { layout: "grid", cols: 3, rows: 2, frameMode: "tiles", duotone: true, duoIdx: 1, lookMode: "cycle", look: 6,
    detail: 120, contrast: 1.35, matte: "#05060E", gap: 6, radius: 2, borderW: 0, shadow: 0.5, rot: 0,
    vignette: 0.5, scan: 0.28, grain: 0.08, backdrop: 0.12 },
  y2k: { layout: "polaroid", cols: 3, rows: 2, frameMode: "same", duotone: true, duoIdx: 13, lookMode: "cycle", look: 3,
    detail: 90, contrast: 1.15, matte: "#FF9AD5", gap: 10, radius: 10, borderW: 10, border: "#FFFFFF", shadow: 0.45,
    jitter: 0.85, rot: 0.7, vignette: 0.08, scan: 0, grain: 0.05, backdrop: 0 },
  brutalist: { layout: "grid", cols: 2, rows: 2, frameMode: "tiles", duotone: false, lookMode: "same", look: 2,
    detail: 100, contrast: 1.65, matte: "#C9C6BF", gap: 26, radius: 0, borderW: 0, shadow: 0, rot: 0,
    vignette: 0, scan: 0, grain: 0.12, backdrop: 0 },
  editorial: { layout: "hero", cols: 2, rows: 2, frameMode: "same", duotone: false, lookMode: "cycle", look: 8,
    detail: 110, contrast: 1.1, matte: "#F4F2ED", gap: 18, radius: 0, borderW: 0, shadow: 0.18, rot: 0,
    vignette: 0.05, scan: 0, grain: 0.04, backdrop: 0 },
  glitch: { layout: "grid", cols: 3, rows: 3, frameMode: "echo", duotone: true, duoIdx: 0, lookMode: "cycle", look: 6,
    detail: 100, contrast: 1.3, matte: "#060608", gap: 3, radius: 0, borderW: 0, shadow: 0.3, rot: 0,
    vignette: 0.35, scan: 0.2, grain: 0.15, backdrop: 0 },
  japanese: { layout: "grid", cols: 4, rows: 2, frameMode: "tiles", duotone: false, lookMode: "same", look: 4,
    detail: 130, contrast: 1.25, matte: "#EDEDEA", gap: 12, radius: 0, borderW: 1, border: "#1A1A1A", shadow: 0.1, rot: 0,
    vignette: 0, scan: 0, grain: 0.03, backdrop: 0 },
  retro: { layout: "grid", cols: 2, rows: 2, frameMode: "same", duotone: true, duoIdx: 8, lookMode: "same", look: 0,
    detail: 110, contrast: 1.3, matte: "#02120A", gap: 8, radius: 3, borderW: 0, shadow: 0.3, rot: 0,
    vignette: 0.45, scan: 0.35, grain: 0.06, backdrop: 0 },
  fashion: { layout: "hero", cols: 2, rows: 2, frameMode: "same", duotone: false, lookMode: "cycle", look: 8,
    detail: 120, contrast: 1.15, matte: "#0E0E0E", gap: 14, radius: 0, borderW: 6, border: "#FFFFFF", shadow: 0.5, rot: 0,
    vignette: 0.3, scan: 0, grain: 0.08, backdrop: 0 },
  blueprint: { layout: "grid", cols: 3, rows: 2, frameMode: "tiles", duotone: true, duoIdx: 9, lookMode: "same", look: 4,
    detail: 140, contrast: 1.45, matte: "#061423", gap: 8, radius: 0, borderW: 1, border: "#2E5C86", shadow: 0.2, rot: 0,
    vignette: 0.3, scan: 0.1, grain: 0.05, backdrop: 0 },
  scrapbook: { layout: "polaroid", cols: 3, rows: 2, frameMode: "same", duotone: false, lookMode: "cycle", look: 9,
    detail: 90, contrast: 1.1, matte: "#6E5844", gap: 10, radius: 3, borderW: 12, border: "#FFFDF5", shadow: 0.55,
    jitter: 0.9, rot: 0.8, vignette: 0.25, scan: 0, grain: 0.16, backdrop: 0 },
  terminal: { layout: "grid", cols: 2, rows: 2, frameMode: "dolly", duotone: true, duoIdx: 11, lookMode: "same", look: 0,
    detail: 120, contrast: 1.3, matte: "#0B0906", gap: 4, radius: 0, borderW: 0, shadow: 0.25, rot: 0,
    vignette: 0.4, scan: 0.3, grain: 0.07, backdrop: 0 },
  album: { layout: "grid", cols: 1, rows: 1, frameMode: "same", duotone: true, duoIdx: 2, lookMode: "same", look: 1,
    detail: 96, contrast: 1.3, matte: "#0A0A0A", gap: 34, radius: 0, borderW: 0, shadow: 0.6, rot: 0,
    vignette: 0.55, scan: 0, grain: 0.2, backdrop: 0 },
  ransom: { layout: "grid", cols: 3, rows: 3, frameMode: "tiles", duotone: false, lookMode: "cycle", look: 0,
    detail: 80, contrast: 1.4, matte: "#111111", gap: 10, radius: 2, borderW: 6, border: "#FFFFFF", shadow: 0.45,
    rot: 0.95, vignette: 0.2, scan: 0, grain: 0.1, backdrop: 0 },
  vaporwave: { layout: "grid", cols: 3, rows: 2, frameMode: "dolly", duotone: true, duoIdx: 10, lookMode: "cycle", look: 1,
    detail: 110, contrast: 1.2, matte: "#160B2A", gap: 8, radius: 4, borderW: 0, shadow: 0.35, rot: 0,
    vignette: 0.3, scan: 0.18, grain: 0.05, backdrop: 0.1 },
};

const CL_DEFAULTS = {
  layout: "grid",
  frameMode: "same",
  duotone: false,
  duoIdx: 0,
  ramp: false,
  rot: 0,
  cols: 1, rows: 1,
  wedges: 8,
  gap: 10, radius: 6, jitter: 0.45,
  matte: "#0B0D10", border: "#F4F1EA", borderW: 0, shadow: 0.35,
  lookMode: "cycle", look: 0,
  detail: 90, cut: 0.04, contrast: 1.15, brightness: 0, gamma: 1,
  backdrop: 0,
  mode: "live", shots: 4, interval: 700,
  vignette: 0.22, grain: 0.05, scan: 0,
  fit: "cover", mirror: true, zoom: 1, panX: 0, panY: 0,
};

// deterministic jitter so panels don't twitch every frame
function hash01(i, salt) {
  const x = Math.sin(i * 127.1 + salt * 311.7) * 43758.5453;
  return x - Math.floor(x);
}

function collageRects(cfg, W, H) {
  const out = [];
  const g = cfg.gap;
  if (cfg.layout === "strip") {
    const n = Math.max(2, cfg.rows);
    const w = Math.min(W - g * 2, (H - g * (n + 1)) * 1.35);
    const h = (H - g * (n + 1)) / n;
    for (let i = 0; i < n; i++) out.push({ x: (W - w) / 2, y: g + i * (h + g), w, h, rot: 0 });
    return out;
  }
  if (cfg.layout === "split") {
    const n = Math.max(2, cfg.cols);
    const w = (W - g * (n + 1)) / n;
    for (let i = 0; i < n; i++) out.push({ x: g + i * (w + g), y: g, w, h: H - g * 2, rot: 0 });
    return out;
  }
  if (cfg.layout === "polaroid") {
    const n = Math.max(2, Math.min(12, cfg.cols * cfg.rows));
    const cols = Math.max(1, cfg.cols), rows = Math.ceil(n / cols);
    const cw = W / cols, chh = H / rows;
    const size = Math.min(cw, chh) * 0.78;
    for (let i = 0; i < n; i++) {
      const c = i % cols, r = Math.floor(i / cols);
      out.push({
        x: c * cw + (cw - size) / 2 + (hash01(i, 1) - 0.5) * cw * 0.16 * cfg.jitter,
        y: r * chh + (chh - size) / 2 + (hash01(i, 2) - 0.5) * chh * 0.16 * cfg.jitter,
        w: size, h: size,
        rot: (hash01(i, 3) - 0.5) * 0.5 * (cfg.rot || cfg.jitter),
      });
    }
    return out;
  }
  if (cfg.layout === "hero") {
    const n = Math.max(2, Math.min(7, cfg.cols * cfg.rows));
    const bigW = W * 0.63 - g * 1.5, bigH = H - g * 2;
    out.push({ x: g, y: g, w: bigW, h: bigH, rot: 0 });
    const rest = n - 1;
    const colW = W - bigW - g * 3;
    const hh = (bigH - g * (rest - 1)) / rest;
    for (let i = 0; i < rest; i++)
      out.push({ x: bigW + g * 2, y: g + i * (hh + g), w: colW, h: hh, rot: 0 });
    return out;
  }
  const cols = Math.max(1, cfg.cols), rows = Math.max(1, cfg.rows);
  const w = (W - g * (cols + 1)) / cols;
  const h = (H - g * (rows + 1)) / rows;
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      out.push({ x: g + c * (w + g), y: g + r * (h + g), w, h,
        rot: (hash01(i, 4) - 0.5) * 0.42 * cfg.rot });
    }
  return out;
}

// Which slice of the source a given panel shows.
function panelCrop(cfg, i, n) {
  if (cfg.frameMode === "tiles") {
    const cols = cfg.layout === "grid" || cfg.layout === "polaroid" ? Math.max(1, cfg.cols)
      : cfg.layout === "split" ? Math.max(1, cfg.cols) : 1;
    const rows = Math.max(1, Math.ceil(n / cols));
    const c = i % cols, r = Math.floor(i / cols);
    return { x: c / cols, y: r / rows, w: 1 / cols, h: 1 / rows };
  }
  if (cfg.frameMode === "dolly") {
    const k = 1 / (1 + (i / Math.max(1, n - 1)) * 2.4);
    return { x: (1 - k) / 2, y: (1 - k) / 2, w: k, h: k };
  }
  return { x: 0, y: 0, w: 1, h: 1 };
}

// Crop plus the global zoom and pan, resolved into source pixels.
function sourceRegion(el, crop, cfg) {
  const sw = el.videoWidth || el.naturalWidth || el.width || 1;
  const sh = el.videoHeight || el.naturalHeight || el.height || 1;
  const z = 1 / Math.max(0.05, cfg.zoom);
  const w = crop.w * z, h = crop.h * z;
  const x = crop.x + crop.w / 2 - w / 2 - cfg.panX * crop.w;
  const y = crop.y + crop.h / 2 - h / 2 - cfg.panY * crop.h;
  return {
    sx: Math.max(0, Math.min(sw - 1, x * sw)),
    sy: Math.max(0, Math.min(sh - 1, y * sh)),
    sw: Math.max(1, Math.min(sw, w * sw)),
    sh: Math.max(1, Math.min(sh, h * sh)),
  };
}

// Renders one treated panel into an offscreen canvas the size of the panel.
function paintPanel(E, srcEl, pw, ph, look, cfg, px, opt) {
  const sw = srcEl.videoWidth || srcEl.naturalWidth || srcEl.width || 0;
  const sh = srcEl.videoHeight || srcEl.naturalHeight || srcEl.height || 0;
  if (!sw || !sh || pw < 2 || ph < 2) return null;
  const reg = (opt && opt.region) || { sx: 0, sy: 0, sw, sh };
  const duo = opt && opt.duo;
  const detailMul = (opt && opt.detailMul) || 1;

  const cw = Math.max(2, Math.round(pw * px)), chh = Math.max(2, Math.round(ph * px));
  const panel = scratch(E, "clpanel", cw, chh);
  const pc = panel.getContext("2d");
  pc.setTransform(px, 0, 0, px, 0, 0);
  pc.globalCompositeOperation = "source-over";
  pc.globalAlpha = 1;
  pc.filter = "none";
  pc.fillStyle = duo ? duo[1] : look.bg;
  pc.fillRect(0, 0, pw, ph);

  const drawSource = (ctx2, w2, h2, alpha) => {
    const k = cfg.fit === "contain" ? Math.min(w2 / reg.sw, h2 / reg.sh) : Math.max(w2 / reg.sw, h2 / reg.sh);
    const dw = reg.sw * k, dh = reg.sh * k;
    ctx2.save();
    ctx2.globalAlpha = alpha;
    if (cfg.mirror) { ctx2.translate(w2, 0); ctx2.scale(-1, 1); }
    try { ctx2.drawImage(srcEl, reg.sx, reg.sy, reg.sw, reg.sh, (w2 - dw) / 2, (h2 - dh) / 2, dw, dh); } catch {}
    ctx2.restore();
  };

  if (look.style === "photo") {
    drawSource(pc, pw, ph, 1);
    if (look.grade !== "none" || duo) {
      // grade a plain photo panel by re-reading it once
      const g = scratch(E, "clgrade", cw, chh);
      const gc = g.getContext("2d", { willReadFrequently: true });
      gc.setTransform(1, 0, 0, 1, 0, 0);
      gc.globalCompositeOperation = "copy";
      gc.drawImage(panel, 0, 0);
      const img = gc.getImageData(0, 0, cw, chh);
      const d = img.data;
      const fn = GRADES[look.grade] || ((r, g, b) => [r, g, b]);
      if (duo) {
        const A = hexToRgb(duo[0]), B = hexToRgb(duo[1]);
        for (let i = 0; i < d.length; i += 4) {
          const v = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255;
          d[i] = B.r + (A.r - B.r) * v;
          d[i + 1] = B.g + (A.g - B.g) * v;
          d[i + 2] = B.b + (A.b - B.b) * v;
        }
        gc.putImageData(img, 0, 0);
        pc.setTransform(1, 0, 0, 1, 0, 0);
        pc.drawImage(g, 0, 0);
        pc.setTransform(px, 0, 0, px, 0, 0);
        return panel;
      }
      for (let i = 0; i < d.length; i += 4) {
        const c = fn(d[i], d[i + 1], d[i + 2]);
        d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2];
      }
      gc.putImageData(img, 0, 0);
      pc.setTransform(1, 0, 0, 1, 0, 0);
      pc.drawImage(g, 0, 0);
      pc.setTransform(px, 0, 0, px, 0, 0);
    }
    return panel;
  }

  if (cfg.backdrop > 0.001) drawSource(pc, pw, ph, cfg.backdrop);

  const glyph = GLYPH_STYLES.has(look.style);
  const cols = Math.max(10, Math.round(cfg.detail * detailMul));
  const cellW = pw / cols;
  pc.font = `100px ${MONO}`;
  const adv = pc.measureText("M").width / 100 || 0.6;
  const fontSize = cellW / adv;
  const cellH = glyph ? fontSize : cellW;
  const rows = Math.max(4, Math.floor(ph / cellH));
  const yPad = (ph - rows * cellH) / 2;
  const sx = look.style === "braille" ? 2 : 1;
  const sy = look.style === "braille" ? 4 : look.style === "blocks" ? 2 : 1;
  const gw = cols * sx, gh = rows * sy;

  const sc = scratch(E, "clsc", gw, gh);
  const sctx = sc.getContext("2d", { willReadFrequently: true });
  sctx.setTransform(1, 0, 0, 1, 0, 0);
  sctx.clearRect(0, 0, gw, gh);
  sctx.imageSmoothingEnabled = true;
  sctx.imageSmoothingQuality = "high";
  if (cfg.mirror) { sctx.translate(gw, 0); sctx.scale(-1, 1); }
  const ga = (reg.sw / reg.sh) * (cellH / cellW) * (sx / sy);
  const gridA = gw / gh;
  let dw2, dh2;
  if (cfg.fit === "cover" ? ga > gridA : ga < gridA) { dh2 = gh; dw2 = gh * ga; }
  else { dw2 = gw; dh2 = gw / ga; }
  try {
    sctx.drawImage(srcEl, reg.sx, reg.sy, reg.sw, reg.sh, (gw - dw2) / 2, (gh - dh2) / 2, dw2, dh2);
  } catch { return panel; }
  const data = sctx.getImageData(0, 0, gw, gh).data;

  const n = gw * gh;
  if (!E.cllum || E.cllum.length !== n) {
    E.cllum = new Float32Array(n);
    E.clhit = new Uint8Array(n);
    E.clrgb = new Uint8ClampedArray(n * 3);
  }
  const lum = E.cllum, hit = E.clhit, rgb = E.clrgb;
  hit.fill(0);
  const grade = GRADES[look.grade];
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    if (data[o + 3] < 8) continue;
    let R = data[o], G = data[o + 1], B = data[o + 2];
    if (grade) { const c2 = grade(R, G, B); R = c2[0]; G = c2[1]; B = c2[2]; }
    let v = (0.299 * R + 0.587 * G + 0.114 * B) / 255;
    v = (v - 0.5) * cfg.contrast + 0.5 + cfg.brightness;
    if (cfg.gamma !== 1) v = Math.pow(Math.max(v, 0), cfg.gamma);
    v = v < 0 ? 0 : v > 1 ? 1 : v;
    if (v < cfg.cut) continue;
    hit[i] = 1; lum[i] = v;
    rgb[i * 3] = R; rgb[i * 3 + 1] = G; rgb[i * 3 + 2] = B;
  }

  const mk = scratch(E, "clmk", cw, chh);
  const m = mk.getContext("2d");
  m.setTransform(px, 0, 0, px, 0, 0);
  m.globalCompositeOperation = "source-over";
  m.clearRect(0, 0, pw, ph);
  m.textBaseline = "top";
  m.textAlign = "left";
  m.font = `${fontSize.toFixed(2)}px ${MONO}`;
  const tinted = look.color !== "mono" || !!duo;
  drawMarks(m, { ...look, dot: 1, shape: "round", depth: 0.6 }, {
    cols, rows, cellW, cellH, yPad, lum, hit,
    flat: tinted ? "#ffffff" : look.tint,
    ramp: RAMPS[look.ramp] || RAMPS.standard,
    at: (x, y) => y * gw + x,
  });

  if (tinted) {
    const cg = scratch(E, "clcg", gw, gh);
    const cgx = cg.getContext("2d");
    if (!E.climg || E.climg.width !== gw || E.climg.height !== gh) E.climg = cgx.createImageData(gw, gh);
    const img = E.climg;
    const cd = img.data;
    if (duo) {
      const A = hexToRgb(duo[0]), B = hexToRgb(duo[1]);
      for (let i = 0; i < n; i++) {
        const o = i * 4, v = lum[i];
        cd[o] = B.r + (A.r - B.r) * v;
        cd[o + 1] = B.g + (A.g - B.g) * v;
        cd[o + 2] = B.b + (A.b - B.b) * v;
        cd[o + 3] = 255;
      }
    } else {
      for (let i = 0; i < n; i++) {
        const o = i * 4;
        cd[o] = rgb[i * 3]; cd[o + 1] = rgb[i * 3 + 1]; cd[o + 2] = rgb[i * 3 + 2]; cd[o + 3] = 255;
      }
    }
    cgx.putImageData(img, 0, 0);
    const keep = scratch(E, "clkeep", cw, chh);
    const kx = keep.getContext("2d");
    kx.setTransform(1, 0, 0, 1, 0, 0);
    kx.globalCompositeOperation = "copy";
    kx.drawImage(mk, 0, 0);
    kx.globalCompositeOperation = "source-over";
    m.setTransform(1, 0, 0, 1, 0, 0);
    m.imageSmoothingEnabled = false;
    m.globalCompositeOperation = "multiply";
    m.drawImage(cg, 0, yPad * px, cw, rows * cellH * px);
    m.globalCompositeOperation = "destination-in";
    m.drawImage(keep, 0, 0);
    m.globalCompositeOperation = "source-over";
    m.imageSmoothingEnabled = true;
  }

  pc.setTransform(1, 0, 0, 1, 0, 0);
  pc.drawImage(mk, 0, 0);
  return panel;
}


function drawCollageFrame(E, cv, cfg, host, shots) {
  const live = E.src && E.src.el;
  if (!cv || !host) return null;
  const W = host.clientWidth, H = host.clientHeight;
  if (W < 8 || H < 8) return null;
  // Every panel runs a full treatment pass, so preview density falls as panels
  // multiply. Export is unaffected — it sets E.cap.
  const panels = cfg.layout === "mirror" || cfg.layout === "kaleido" ? 1
    : cfg.layout === "strip" ? cfg.rows
    : cfg.layout === "split" ? cfg.cols
    : cfg.layout === "hero" ? Math.max(2, Math.min(7, cfg.cols * cfg.rows))
    : cfg.cols * cfg.rows;
  const px = E.cap || (panels > 6 ? 0.75 : panels > 2 ? 1 : Math.min(window.devicePixelRatio || 1, 2));
  let cw = Math.floor(W * px), chh = Math.floor(H * px);
  if (E.cap) { cw -= cw % 2; chh -= chh % 2; }
  if (cv.width !== cw || cv.height !== chh) { cv.width = cw; cv.height = chh; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(px, 0, 0, px, 0, 0);
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
  ctx.filter = "none";
  ctx.fillStyle = cfg.matte;
  if (E.alpha) ctx.clearRect(0, 0, W, H); else ctx.fillRect(0, 0, W, H);
  if (!live && !(shots && shots.length)) return { panels: 0 };

  const lookFor = (i) => LOOKS[(cfg.look + (cfg.lookMode === "same" ? 0 : i)) % LOOKS.length];
  const count = cfg.layout === "mirror" || cfg.layout === "kaleido" ? 1
    : cfg.layout === "strip" ? cfg.rows
    : cfg.layout === "split" ? cfg.cols
    : cfg.layout === "hero" ? Math.max(2, Math.min(7, cfg.cols * cfg.rows))
    : cfg.cols * cfg.rows;

  // Echo: a rolling buffer of recent frames, so panels read as motion trails.
  if (cfg.frameMode === "echo" && live) {
    const need = Math.max(2, count);
    if (!E.echo || E.echo.length !== need) {
      E.echo = Array.from({ length: need }, () => document.createElement("canvas"));
      E.echoHead = 0; E.echoT = 0;
    }
    const now = performance.now();
    if (now - (E.echoT || 0) > Math.max(40, cfg.interval / 4)) {
      E.echoT = now;
      E.echoHead = (E.echoHead + 1) % need;
      const c = E.echo[E.echoHead];
      const vw = live.videoWidth || live.naturalWidth || 640;
      const vh = live.videoHeight || live.naturalHeight || 360;
      const tw = 512, th = Math.max(2, Math.round((512 * vh) / vw));
      if (c.width !== tw || c.height !== th) { c.width = tw; c.height = th; }
      try { c.getContext("2d").drawImage(live, 0, 0, tw, th); } catch {}
    }
  }

  const frameFor = (i) => {
    if (cfg.frameMode === "echo" && E.echo && E.echo.length)
      return E.echo[(E.echoHead - i + E.echo.length * 2) % E.echo.length];
    if (cfg.mode === "time" && shots && shots.length) return shots[i % shots.length];
    return live;
  };
  const optFor = (i, el) => ({
    region: sourceRegion(el, panelCrop(cfg, i, count), cfg),
    duo: cfg.duotone ? DUOS[((cfg.duoIdx || 0) + i) % DUOS.length] : null,
    detailMul: cfg.ramp ? 0.3 + 0.7 * (i / Math.max(1, count - 1)) : 1,
  });

  if (cfg.layout === "mirror" || cfg.layout === "kaleido") {
    const el0 = frameFor(0);
    if (!el0) return { panels: 0 };
    const panel = paintPanel(E, el0, W, H, lookFor(0), cfg, px, optFor(0, el0));
    if (!panel) return { panels: 0 };
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (cfg.layout === "mirror") {
      // one quadrant, then its three reflections
      const hw = cv.width / 2, hh = cv.height / 2;
      ctx.drawImage(panel, 0, 0, hw, hh, 0, 0, hw, hh);
      ctx.save(); ctx.translate(cv.width, 0); ctx.scale(-1, 1);
      ctx.drawImage(panel, 0, 0, hw, hh, 0, 0, hw, hh); ctx.restore();
      ctx.save(); ctx.translate(0, cv.height); ctx.scale(1, -1);
      ctx.drawImage(panel, 0, 0, hw, hh, 0, 0, hw, hh); ctx.restore();
      ctx.save(); ctx.translate(cv.width, cv.height); ctx.scale(-1, -1);
      ctx.drawImage(panel, 0, 0, hw, hh, 0, 0, hw, hh); ctx.restore();
    } else {
      const n = Math.max(3, cfg.wedges);
      const step = (Math.PI * 2) / n;
      const R = Math.hypot(cv.width, cv.height);
      for (let i = 0; i < n; i++) {
        ctx.save();
        ctx.translate(cv.width / 2, cv.height / 2);
        ctx.rotate(i * step);
        if (i % 2) ctx.scale(-1, 1);
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.arc(0, 0, R, -step / 2, step / 2);
        ctx.closePath();
        ctx.clip();
        ctx.drawImage(panel, -cv.width / 2, -cv.height / 2);
        ctx.restore();
      }
    }
  } else {
    const rects = collageRects(cfg, W, H);
    rects.forEach((r, i) => {
      const el = frameFor(i);
      if (!el) return;
      const bw = cfg.borderW;
      const panel = paintPanel(E, el, Math.max(2, r.w - bw * 2), Math.max(2, r.h - bw * 2), lookFor(i), cfg, px, optFor(i, el));
      if (!panel) return;
      ctx.save();
      ctx.setTransform(px, 0, 0, px, 0, 0);
      ctx.translate(r.x + r.w / 2, r.y + r.h / 2);
      if (r.rot) ctx.rotate(r.rot);
      if (cfg.shadow > 0.001) {
        ctx.shadowColor = `rgba(0,0,0,${Math.min(0.8, cfg.shadow)})`;
        ctx.shadowBlur = 10 + cfg.shadow * 26;
        ctx.shadowOffsetY = 3 + cfg.shadow * 8;
      }
      const rr = Math.min(cfg.radius, Math.min(r.w, r.h) / 2);
      const round = (x, y, w, h, rad) => {
        ctx.beginPath();
        ctx.moveTo(x + rad, y);
        ctx.arcTo(x + w, y, x + w, y + h, rad);
        ctx.arcTo(x + w, y + h, x, y + h, rad);
        ctx.arcTo(x, y + h, x, y, rad);
        ctx.arcTo(x, y, x + w, y, rad);
        ctx.closePath();
      };
      if (bw > 0) {
        ctx.fillStyle = cfg.border;
        round(-r.w / 2, -r.h / 2, r.w, r.h, rr);
        ctx.fill();
      }
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;
      ctx.save();
      round(-r.w / 2 + bw, -r.h / 2 + bw, r.w - bw * 2, r.h - bw * 2, Math.max(0, rr - bw));
      ctx.clip();
      ctx.drawImage(panel, -r.w / 2 + bw, -r.h / 2 + bw, r.w - bw * 2, r.h - bw * 2);
      ctx.restore();
      ctx.restore();
    });
  }

  ctx.setTransform(px, 0, 0, px, 0, 0);
  if (cfg.scan > 0.001) {
    ctx.save();
    ctx.globalAlpha = cfg.scan;
    ctx.fillStyle = "#000";
    for (let y = 0; y < H; y += 3) ctx.fillRect(0, y, W, 1.2);
    ctx.restore();
  }
  if (cfg.vignette > 0.001) {
    const g = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.3, W / 2, H / 2, Math.max(W, H) * 0.72);
    g.addColorStop(0, "rgba(0,0,0,0)");
    g.addColorStop(1, `rgba(0,0,0,${Math.min(0.95, cfg.vignette)})`);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
  }
  if (cfg.grain > 0.001) {
    const gs = 220;
    const gn = scratch(E, "clgrain", gs, gs);
    const gc = gn.getContext("2d");
    const img = gc.createImageData(gs, gs);
    const d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      const v = 128 + (Math.random() - 0.5) * 255;
      d[i] = d[i + 1] = d[i + 2] = v; d[i + 3] = 255;
    }
    gc.putImageData(img, 0, 0);
    ctx.save();
    ctx.globalAlpha = Math.min(0.6, cfg.grain);
    ctx.globalCompositeOperation = "overlay";
    for (let y = 0; y < H; y += gs) for (let x = 0; x < W; x += gs) ctx.drawImage(gn, x, y);
    ctx.restore();
  }
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
  return { panels };
}

function drawMediaFrame(E, cv, cfg, host) {
  const el = E.src && E.src.el;
  if (!el || !cv || !host) return null;
  const sw = el.videoWidth || el.naturalWidth || 0;
  const sh = el.videoHeight || el.naturalHeight || 0;
  if (!sw || !sh) return null;

  const W = host.clientWidth, H = host.clientHeight;
  if (W < 8 || H < 8) return null;
  const px = E.cap || Math.min(window.devicePixelRatio || 1, 2);
  let cw = Math.floor(W * px), ch = Math.floor(H * px);
  if (E.cap) { cw -= cw % 2; ch -= ch % 2; }
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(px, 0, 0, px, 0, 0);
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
  ctx.filter = "none";

  const glyph = GLYPH_STYLES.has(cfg.style);
  const cols = Math.max(20, Math.round(cfg.detail));
  const cellW = W / cols;
  ctx.font = `100px ${MONO}`;
  const adv = ctx.measureText("M").width / 100 || 0.6;
  const fontSize = cellW / adv;
  const cellH = glyph ? fontSize : cellW;
  const rows = Math.max(6, Math.floor(H / cellH));
  const yPad = (H - rows * cellH) / 2;

  const sx = cfg.style === "braille" ? 2 : 1;
  const sy = cfg.style === "braille" ? 4 : cfg.style === "blocks" ? 2 : 1;
  const gw = cols * sx, gh = rows * sy;

  const sc = scratch(E, "mdsc", gw, gh);
  const sctx = sc.getContext("2d", { willReadFrequently: true });
  sctx.setTransform(1, 0, 0, 1, 0, 0);
  sctx.clearRect(0, 0, gw, gh);
  sctx.imageSmoothingEnabled = true;
  sctx.imageSmoothingQuality = "high";
  if (cfg.mirror) { sctx.translate(gw, 0); sctx.scale(-1, 1); }
  // fit against the aspect the cell grid actually displays at
  const ga = (sw / sh) * (cellH / cellW) * (sx / sy);
  const grid = gw / gh;
  let dw, dh;
  if (cfg.fit === "cover" ? ga > grid : ga < grid) { dh = gh; dw = gh * ga; }
  else { dw = gw; dh = gw / ga; }
  dw *= cfg.zoom; dh *= cfg.zoom;
  try { sctx.drawImage(el, (gw - dw) / 2 + cfg.panX * gw, (gh - dh) / 2 + cfg.panY * gh, dw, dh); } catch { return null; }
  const data = sctx.getImageData(0, 0, gw, gh).data;

  // tone + grade into luminance and colour tables
  const n = gw * gh;
  if (!E.mdlum || E.mdlum.length !== n) {
    E.mdlum = new Float32Array(n);
    E.mdhit = new Uint8Array(n);
    E.mdrgb = new Uint8ClampedArray(n * 3);
  }
  const lum = E.mdlum, hit = E.mdhit, rgb = E.mdrgb;
  hit.fill(0);
  const grade = GRADES[cfg.grade];
  const cut = cfg.cut;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    if (data[o + 3] < 8) continue;
    let R = data[o], G = data[o + 1], B = data[o + 2];
    if (grade) { const c = grade(R, G, B); R = c[0]; G = c[1]; B = c[2]; }
    if (cfg.sat !== 1) {
      const m = (R + G + B) / 3;
      R = m + (R - m) * cfg.sat; G = m + (G - m) * cfg.sat; B = m + (B - m) * cfg.sat;
    }
    let v = (0.299 * R + 0.587 * G + 0.114 * B) / 255;
    v = (v - 0.5) * cfg.contrast + 0.5 + cfg.brightness;
    if (cfg.gamma !== 1) v = Math.pow(Math.max(v, 0), cfg.gamma);
    if (cfg.invert) v = 1 - v;
    v = v < 0 ? 0 : v > 1 ? 1 : v;
    if (v < cut) continue;
    hit[i] = 1;
    lum[i] = v;
    rgb[i * 3] = R; rgb[i * 3 + 1] = G; rgb[i * 3 + 2] = B;
  }

  /* ---- backdrop ---- */
  ctx.fillStyle = cfg.bg;
  if (E.alpha) ctx.clearRect(0, 0, W, H); else ctx.fillRect(0, 0, W, H);
  if (cfg.backdrop === "orig" || cfg.backdrop === "blur") {
    let [bx, by, bw, bh] = coverRect(sw, sh, W, H, "cover");
    bw *= cfg.zoom; bh *= cfg.zoom;
    bx = (W - bw) / 2 + cfg.panX * W;
    by = (H - bh) / 2 + cfg.panY * H;
    ctx.save();
    ctx.globalAlpha = cfg.backOpacity;
    if (cfg.backdrop === "blur") ctx.filter = `blur(${cfg.blur}px)`;
    if (cfg.mirror) { ctx.translate(W, 0); ctx.scale(-1, 1); }
    try { ctx.drawImage(el, bx, by, bw, bh); } catch {}
    ctx.restore();
    ctx.filter = "none";
    ctx.globalAlpha = 1;
  }

  /* ---- marks on their own layer ---- */
  const mk = scratch(E, "mdmk", cv.width, cv.height);
  const m = mk.getContext("2d");
  m.setTransform(px, 0, 0, px, 0, 0);
  m.globalCompositeOperation = "source-over";
  m.clearRect(0, 0, W, H);
  m.textBaseline = "top";
  m.textAlign = "left";
  m.font = `${fontSize.toFixed(2)}px ${MONO}`;
  const ramp = (cfg.ramp === "custom" ? cfg.custom : RAMPS[cfg.ramp]) || RAMPS.standard;
  const tinted = cfg.color !== "mono";
  const text = drawMarks(m, cfg, {
    cols, rows, cellW, cellH, yPad, lum, hit,
    flat: tinted ? "#ffffff" : cfg.tint, ramp, at: (x, y) => y * gw + x,
  });

  // Per-cell colour in one composite: multiply the sampled grid over the flat
  // marks, then restore the mark alpha. Constant cost, whatever the cell count.
  if (tinted) {
    const cg = scratch(E, "mdcg", gw, gh);
    const cgx = cg.getContext("2d");
    if (!E.mdimg || E.mdimg.width !== gw || E.mdimg.height !== gh) E.mdimg = cgx.createImageData(gw, gh);
    const cd = E.mdimg.data;
    for (let i = 0; i < n; i++) {
      const o = i * 4;
      cd[o] = rgb[i * 3]; cd[o + 1] = rgb[i * 3 + 1]; cd[o + 2] = rgb[i * 3 + 2];
      cd[o + 3] = 255;
    }
    cgx.putImageData(E.mdimg, 0, 0);
    const keep = scratch(E, "mdkeep", cv.width, cv.height);
    const kx = keep.getContext("2d");
    kx.setTransform(1, 0, 0, 1, 0, 0);
    kx.globalCompositeOperation = "copy";
    kx.drawImage(mk, 0, 0);
    kx.globalCompositeOperation = "source-over";
    m.setTransform(1, 0, 0, 1, 0, 0);
    m.imageSmoothingEnabled = false;
    m.globalCompositeOperation = "multiply";
    m.drawImage(cg, 0, yPad * px, cv.width, rows * cellH * px);
    m.globalCompositeOperation = "destination-in";
    m.drawImage(keep, 0, 0);
    m.globalCompositeOperation = "source-over";
    m.imageSmoothingEnabled = true;
  }

  /* ---- character-level effects, before the marks meet the backdrop ---- */
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const tintCanvas = () => scratch(E, "mdtint", cv.width, cv.height);
  const splitDraw = (srcCanvas, amount) => {
    const d = amount * 8 * px;
    const tint = tintCanvas();
    const t = tint.getContext("2d");
    ctx.globalCompositeOperation = "lighter";
    for (const [col, off] of [["#ff2040", -d], ["#20ff80", 0], ["#2060ff", d]]) {
      t.setTransform(1, 0, 0, 1, 0, 0);
      t.globalCompositeOperation = "source-over";
      t.clearRect(0, 0, cv.width, cv.height);
      t.drawImage(srcCanvas, 0, 0);
      t.globalCompositeOperation = "source-in";
      t.fillStyle = col;
      t.fillRect(0, 0, cv.width, cv.height);
      ctx.drawImage(tint, off, 0);
    }
    ctx.globalCompositeOperation = "source-over";
  };

  if (cfg.cChroma > 0.001) splitDraw(mk, cfg.cChroma);
  else ctx.drawImage(mk, 0, 0);

  if (cfg.cBloom > 0.001) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = Math.min(1, cfg.cBloom);
    ctx.filter = `blur(${(4 + cfg.cBloom * 14) * px}px)`;
    ctx.drawImage(mk, 0, 0);
    ctx.restore();
  }

  /* ---- whole-frame effects ---- */
  const copy = () => {
    const c2 = scratch(E, "mdcopy", cv.width, cv.height);
    const g2 = c2.getContext("2d");
    g2.setTransform(1, 0, 0, 1, 0, 0);
    g2.globalCompositeOperation = "copy";
    g2.drawImage(cv, 0, 0);
    g2.globalCompositeOperation = "source-over";
    return c2;
  };

  if (cfg.pixel > 0.001) {
    const f = Math.max(2, Math.round(2 + cfg.pixel * 46));
    const small = scratch(E, "mdpix", Math.max(2, Math.ceil(cv.width / f)), Math.max(2, Math.ceil(cv.height / f)));
    const sg = small.getContext("2d");
    sg.setTransform(1, 0, 0, 1, 0, 0);
    sg.clearRect(0, 0, small.width, small.height);
    sg.drawImage(cv, 0, 0, small.width, small.height);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(small, 0, 0, cv.width, cv.height);
    ctx.imageSmoothingEnabled = true;
  }

  if (cfg.halftone > 0.001) {
    const pitch = (4 + (1 - cfg.halftone) * 10) * px;
    const hw = Math.max(2, Math.ceil(cv.width / pitch)), hh = Math.max(2, Math.ceil(cv.height / pitch));
    const small = scratch(E, "mdht", hw, hh);
    const sg = small.getContext("2d", { willReadFrequently: true });
    sg.setTransform(1, 0, 0, 1, 0, 0);
    sg.drawImage(cv, 0, 0, hw, hh);
    const hd = sg.getImageData(0, 0, hw, hh).data;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.save();
    ctx.globalCompositeOperation = "multiply";
    ctx.globalAlpha = Math.min(1, cfg.halftone * 1.2);
    ctx.fillStyle = "#000";
    ctx.beginPath();
    for (let y = 0; y < hh; y++)
      for (let x = 0; x < hw; x++) {
        const i = (y * hw + x) * 4;
        const l = (0.299 * hd[i] + 0.587 * hd[i + 1] + 0.114 * hd[i + 2]) / 255;
        const rad = (1 - l) * pitch * 0.72;
        if (rad < 0.25) continue;
        const ax = (x + 0.5) * pitch, ay = (y + 0.5) * pitch;
        ctx.moveTo(ax + rad, ay);
        ctx.arc(ax, ay, rad, 0, Math.PI * 2);
      }
    ctx.fill();
    ctx.restore();
  }

  if (cfg.aberr > 0.001) {
    // radial fringing: the channels disagree about scale, not position
    const c2 = copy();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = "copy";
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.globalCompositeOperation = "lighter";
    const tint = tintCanvas();
    const t = tint.getContext("2d");
    const k = cfg.aberr * 0.012;
    for (const [col, sc2] of [["#ff2040", 1 + k], ["#20ff80", 1], ["#2060ff", 1 - k]]) {
      t.setTransform(1, 0, 0, 1, 0, 0);
      t.globalCompositeOperation = "source-over";
      t.clearRect(0, 0, cv.width, cv.height);
      t.drawImage(c2, 0, 0);
      t.globalCompositeOperation = "source-in";
      t.fillStyle = col;
      t.fillRect(0, 0, cv.width, cv.height);
      t.globalCompositeOperation = "multiply";
      t.drawImage(c2, 0, 0);
      const w2 = cv.width * sc2, h2 = cv.height * sc2;
      ctx.drawImage(tint, (cv.width - w2) / 2, (cv.height - h2) / 2, w2, h2);
    }
    ctx.globalCompositeOperation = "source-over";
  }

  if (cfg.bloom > 0.001) {
    const c2 = copy();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = Math.min(1, cfg.bloom * 0.8);
    ctx.filter = `blur(${(6 + cfg.bloom * 22) * px}px)`;
    ctx.drawImage(c2, 0, 0);
    ctx.restore();
  }

  if (cfg.glitch > 0.001) {
    const c2 = copy();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const slices = Math.round(3 + cfg.glitch * 22);
    for (let i = 0; i < slices; i++) {
      const y = Math.random() * cv.height;
      const h2 = Math.max(2, Math.random() * cv.height * 0.06 * (0.4 + cfg.glitch));
      const off = (Math.random() - 0.5) * cfg.glitch * cv.width * 0.16;
      ctx.drawImage(c2, 0, y, cv.width, h2, off, y, cv.width, h2);
    }
  }

  if (cfg.scan > 0.001) {
    ctx.setTransform(px, 0, 0, px, 0, 0);
    ctx.save();
    ctx.globalAlpha = cfg.scan;
    ctx.fillStyle = "#000";
    for (let y = 0; y < H; y += 3) ctx.fillRect(0, y, W, 1.2);
    ctx.restore();
  }

  if (cfg.crt > 0.001) {
    // barrel bulge approximated per strip, plus a rounded bezel mask
    const c2 = copy();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, cv.width, cv.height);
    const strips = 96, k = cfg.crt * 0.16;
    for (let i = 0; i < strips; i++) {
      const sy2 = (i / strips) * cv.height;
      const sh2 = cv.height / strips + 1;
      const ny = ((i + 0.5) / strips) * 2 - 1;
      const dw2 = cv.width * (1 - k * ny * ny);
      const dy2 = cv.height / 2 + (sy2 + sh2 / 2 - cv.height / 2) * (1 - k * 0.25) - sh2 / 2;
      ctx.drawImage(c2, 0, sy2, cv.width, sh2, (cv.width - dw2) / 2, dy2, dw2, sh2);
    }
    const rr = Math.min(cv.width, cv.height) * 0.06 * (0.5 + cfg.crt);
    ctx.globalCompositeOperation = "destination-in";
    ctx.beginPath();
    ctx.moveTo(rr, 0);
    ctx.arcTo(cv.width, 0, cv.width, cv.height, rr);
    ctx.arcTo(cv.width, cv.height, 0, cv.height, rr);
    ctx.arcTo(0, cv.height, 0, 0, rr);
    ctx.arcTo(0, 0, cv.width, 0, rr);
    ctx.closePath();
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";
  }

  ctx.setTransform(px, 0, 0, px, 0, 0);
  if (cfg.vignette > 0.001) {
    const g = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.28, W / 2, H / 2, Math.max(W, H) * 0.72);
    g.addColorStop(0, "rgba(0,0,0,0)");
    g.addColorStop(1, `rgba(0,0,0,${Math.min(0.96, cfg.vignette)})`);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
  }

  if (cfg.grain > 0.001) {
    const gs = 220;
    const gn = scratch(E, "mdgrain", gs, gs);
    const gc = gn.getContext("2d");
    const img = gc.createImageData(gs, gs);
    const p2 = img.data;
    for (let i = 0; i < p2.length; i += 4) {
      const v = 128 + (Math.random() - 0.5) * 255;
      p2[i] = p2[i + 1] = p2[i + 2] = v;
      p2[i + 3] = 255;
    }
    gc.putImageData(img, 0, 0);
    ctx.save();
    ctx.globalAlpha = Math.min(0.6, cfg.grain);
    ctx.globalCompositeOperation = "overlay";
    for (let y = 0; y < H; y += gs) for (let x = 0; x < W; x += gs) ctx.drawImage(gn, x, y);
    ctx.restore();
  }

  if (cfg.dust > 0.001) {
    ctx.save();
    const specks = Math.round(cfg.dust * 90);
    for (let i = 0; i < specks; i++) {
      ctx.globalAlpha = 0.25 + Math.random() * 0.5;
      ctx.fillStyle = Math.random() > 0.45 ? "#fff" : "#000";
      ctx.beginPath();
      ctx.arc(Math.random() * W, Math.random() * H, Math.random() * 1.6 + 0.3, 0, Math.PI * 2);
      ctx.fill();
    }
    for (let i = 0; i < Math.round(cfg.dust * 4); i++) {
      const x = Math.random() * W;
      ctx.globalAlpha = 0.06 + Math.random() * 0.14;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 0.6 + Math.random();
      ctx.beginPath();
      ctx.moveTo(x, Math.random() * H * 0.3);
      ctx.lineTo(x + (Math.random() - 0.5) * 6, H * (0.5 + Math.random() * 0.5));
      ctx.stroke();
    }
    ctx.restore();
  }

  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
  return { cols, rows, text };
}


/* ============================================================
   MP4 muxing — one AVC track, written by hand.
   MediaRecorder emits files with no duration in the header, which is
   why players refuse to scrub them. Encoding with WebCodecs and writing
   the container ourselves gives an exact duration and a seek index.
   Layout: ftyp | mdat | moov  (moov last, so chunk offsets are known)
   ============================================================ */

const _te = new TextEncoder();
const _s = (x) => _te.encode(x);
const _u16 = (n) => { const b = new Uint8Array(2); new DataView(b.buffer).setUint16(0, n); return b; };
const _u32 = (n) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, n >>> 0); return b; };
const _fl = (v, f) => new Uint8Array([v, (f >> 16) & 255, (f >> 8) & 255, f & 255]);
const _z = (n) => new Uint8Array(n);
const _MATRIX = new Uint8Array([0,1,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0x40,0,0,0]);

function _box(type, ...parts) {
  let len = 8;
  for (const p of parts) len += p.length;
  const b = new Uint8Array(len);
  new DataView(b.buffer).setUint32(0, len);
  b.set(_s(type), 4);
  let o = 8;
  for (const p of parts) { b.set(p, o); o += p.length; }
  return b;
}

function muxMP4({ samples, description, width, height, fps }) {
  if (!samples.length) throw new Error("the encoder produced no frames");
  if (!description) throw new Error("the encoder gave no codec description");
  const timescale = 90000;
  const delta = Math.round(timescale / fps);
  const duration = delta * samples.length;
  let mdatSize = 0;
  for (const smp of samples) mdatSize += smp.data.length;

  const ftyp = _box("ftyp", _s("isom"), _u32(0x200), _s("isom"), _s("iso2"), _s("avc1"), _s("mp41"));
  const dataStart = ftyp.length + 8;

  const avc1 = _box("avc1", _z(6), _u16(1), _u16(0), _u16(0), _z(12),
    _u16(width), _u16(height), _u32(0x00480000), _u32(0x00480000), _u32(0), _u16(1),
    _z(32), _u16(0x0018), new Uint8Array([0xff, 0xff]), _box("avcC", description));

  const keys = [];
  samples.forEach((smp, i) => { if (smp.key) keys.push(i + 1); });
  const stss = keys.length && keys.length !== samples.length
    ? _box("stss", _fl(0, 0), _u32(keys.length), ...keys.map(_u32)) : _z(0);

  const stbl = _box("stbl",
    _box("stsd", _fl(0, 0), _u32(1), avc1),
    _box("stts", _fl(0, 0), _u32(1), _u32(samples.length), _u32(delta)),
    stss,
    _box("stsc", _fl(0, 0), _u32(1), _u32(1), _u32(samples.length), _u32(1)),
    _box("stsz", _fl(0, 0), _u32(0), _u32(samples.length), ...samples.map((smp) => _u32(smp.data.length))),
    _box("stco", _fl(0, 0), _u32(1), _u32(dataStart)));

  const minf = _box("minf", _box("vmhd", _fl(0, 1), _z(8)),
    _box("dinf", _box("dref", _fl(0, 0), _u32(1), _box("url ", _fl(0, 1)))), stbl);
  const mdia = _box("mdia",
    _box("mdhd", _fl(0, 0), _u32(0), _u32(0), _u32(timescale), _u32(duration), _u16(0x55c4), _u16(0)),
    _box("hdlr", _fl(0, 0), _u32(0), _s("vide"), _z(12), _s("Glyphworks\0")), minf);
  const trak = _box("trak",
    _box("tkhd", _fl(0, 3), _u32(0), _u32(0), _u32(1), _u32(0), _u32(duration),
      _z(8), _u16(0), _u16(0), _u16(0), _u16(0), _MATRIX,
      _u32(width * 65536), _u32(height * 65536)), mdia);
  const moov = _box("moov",
    _box("mvhd", _fl(0, 0), _u32(0), _u32(0), _u32(timescale), _u32(duration),
      _u32(0x00010000), _u16(0x0100), _u16(0), _z(8), _MATRIX, _z(24), _u32(2)), trak);

  const out = new Uint8Array(ftyp.length + 8 + mdatSize + moov.length);
  let o = 0;
  out.set(ftyp, o); o += ftyp.length;
  new DataView(out.buffer).setUint32(o, 8 + mdatSize);
  out.set(_s("mdat"), o + 4);
  o += 8;
  for (const smp of samples) { out.set(smp.data, o); o += smp.data.length; }
  out.set(moov, o);
  return out;
}

// H.264 level has to cover the frame size or the encoder refuses the config.
function avcCodec(w, h) {
  const mb = Math.ceil(w / 16) * Math.ceil(h / 16);
  if (mb <= 3600) return "avc1.42001f";   // 3.1, up to 720p
  if (mb <= 8192) return "avc1.420028";   // 4.0, up to 1080p
  if (mb <= 22080) return "avc1.420032";  // 5.0, up to 1440p
  return "avc1.420034";                   // 5.2, 4K
}


/* ============================================================
   Hand control — real hand landmarks, loaded only when switched on.
   Colour and background tricks all failed for the same reason: a person is
   the largest thing in frame that is not the room, so they win every time.
   A landmark model finds the hand specifically and ignores faces and walls.
   The download happens on first use, so page load is untouched.
   ============================================================ */

const MP_VERSION = "0.10.14";
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MP_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Landmarks: 0 wrist, 4 thumb tip, 8 index tip, 9 middle knuckle,
// 12 middle tip, 16 ring tip, 20 little tip.
const TIPS = [8, 12, 16, 20];
const BONES = [
  [0,1],[1,2],[2,3],[3,4], [0,5],[5,6],[6,7],[7,8], [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16], [13,17],[17,18],[18,19],[19,20], [0,17],
];

const HC_DEFAULTS = {
  on: false,
  rotate: true, scatter: true, scale: true,
  sensitivity: 1,
  smooth: 0.75,
  mirror: true,
};

async function loadHandModel(E) {
  if (E.mpLoading) return E.mpLoading;
  E.mpLoading = (async () => {
    const vision = await import(/* @vite-ignore */ `${MP_BASE}`);
    const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
    const landmarker = await vision.HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MP_MODEL, delegate: "GPU" },
      numHands: 1,
      runningMode: "VIDEO",
      minHandDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    E.mp = landmarker;
    return landmarker;
  })();
  return E.mpLoading;
}

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function readHand(E, cfg) {
  const el = E.ctrl && E.ctrl.el;
  if (!el || !el.videoWidth || !E.mp) return null;
  const now = performance.now();
  if (E.mpLast === el.currentTime) return E.hand || null;
  E.mpLast = el.currentTime;

  let res;
  try { res = E.mp.detectForVideo(el, now); } catch { return E.hand || null; }
  const state = E.hand || (E.hand = { x: 0.5, open: 0.5, pinch: 0.5, seen: false, pts: null });
  const lm = res && res.landmarks && res.landmarks[0];
  if (!lm) { state.seen = false; state.pts = null; return state; }

  const palm = Math.max(0.02, dist(lm[0], lm[9]));
  // fingers folded in versus reaching out, measured against the palm so it
  // does not change with distance from the camera
  let reach = 0;
  for (const t of TIPS) reach += dist(lm[t], lm[0]) / palm;
  reach /= TIPS.length;
  const openness = Math.min(1, Math.max(0, (reach - 1.25) / 1.05));
  // thumb tip to index tip, again relative to palm size
  const pinch = Math.min(1, Math.max(0, (dist(lm[4], lm[8]) / palm - 0.25) / 1.25));

  let nx = lm[9].x;
  if (cfg.mirror) nx = 1 - nx;

  const k = Math.min(0.95, Math.max(0, cfg.smooth));
  state.x += (nx - state.x) * (1 - k);
  state.open += (openness - state.open) * (1 - k);
  state.pinch += (pinch - state.pinch) * (1 - k);
  state.seen = true;
  state.pts = lm;
  return state;
}

function applyHand(state, cfg, orbit, cfgRef) {
  if (!state || !state.seen) return;
  const g = Math.max(0.2, cfg.sensitivity);
  let next = null;
  const set = (key, v) => {
    if (!(key in cfgRef.current)) return;
    if (!next) next = { ...cfgRef.current };
    next[key] = v;
  };
  if (cfg.rotate) orbit.yaw = (state.x - 0.5) * Math.PI * 2 * g;
  if (cfg.scatter) set("scatter", state.open * 1.1 * g);
  if (cfg.scale) set("scale", 0.5 + state.pinch * 1.6 * g);
  if (next) cfgRef.current = next;
}


/* ============================================================
   Plotter output — hidden-line removed SVG.
   A pen plotter can only draw lines; it cannot paint faces over lines to
   hide them. So occluded edges have to be removed before export, or the
   drawing comes out as a transparent tangle. Each edge is walked in small
   steps and tested against a depth pass of the solid model, and only the
   visible runs are written out.
   ============================================================ */

const PAPERS = {
  A6: [105, 148], A5: [148, 210], A4: [210, 297], A3: [297, 420], square: [200, 200],
};

const PLOT_DEFAULTS = {
  paper: "A4", landscape: false, margin: 12,
  pen: 0.35, hlr: true, tolerance: 4, hidden: false, quality: 1400,
  fit: true,       // scale the drawing to the page instead of the viewport
  minLen: 0.6,     // drop specks shorter than this, in mm
  join: true,      // chain touching paths so the pen stays down
};

const DEPTH_VS = `
varying float vZ;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vZ = -mv.z;
  gl_Position = projectionMatrix * mv;
}`;

// 16 bits of depth packed across two channels: 8 is not enough to tell an
// edge from the face it sits on.
const DEPTH_FS = `
varying float vZ;
uniform float uNear;
uniform float uFar;
void main() {
  float v = clamp((vZ - uNear) / (uFar - uNear), 0.0, 1.0);
  gl_FragColor = vec4(floor(v * 255.0) / 255.0, fract(v * 255.0), 0.0, 1.0);
}`;

function depthPass(E, w, h) {
  const cam = E.camera;
  if (!E.plotMat)
    E.plotMat = new THREE.ShaderMaterial({
      vertexShader: DEPTH_VS,
      fragmentShader: DEPTH_FS,
      uniforms: { uNear: { value: 0.1 }, uFar: { value: 100 } },
      side: THREE.DoubleSide,
    });
  E.plotMat.uniforms.uNear.value = cam.near;
  E.plotMat.uniforms.uFar.value = cam.far;

  if (!E.plotRT || E.plotRT.width !== w || E.plotRT.height !== h) {
    if (E.plotRT) E.plotRT.dispose();
    E.plotRT = new THREE.WebGLRenderTarget(w, h, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
    });
    E.plotBuf = new Uint8Array(w * h * 4);
  }

  // the solid model has to be visible for the depth pass, the lines must not
  const wasContent = E.content ? E.content.visible : false;
  const wasWire = E.wire ? E.wire.visible : false;
  if (E.content) E.content.visible = true;
  if (E.wire) E.wire.visible = false;
  const fog = E.scene.fog;
  E.scene.fog = null;
  E.scene.overrideMaterial = E.plotMat;
  E.renderer.setRenderTarget(E.plotRT);
  E.renderer.setClearColor(0xffffff, 1); // empty space is infinitely far
  E.renderer.clear();
  E.renderer.render(E.scene, cam);
  E.renderer.setRenderTarget(null);
  E.renderer.readRenderTargetPixels(E.plotRT, 0, 0, w, h, E.plotBuf);
  E.scene.overrideMaterial = null;
  E.scene.fog = fog;
  if (E.content) E.content.visible = wasContent;
  if (E.wire) E.wire.visible = wasWire;
  return E.plotBuf;
}

// Every wireframe edge, in world space.
function collectEdges(wire) {
  const out = [];
  wire.updateMatrixWorld(true);
  wire.traverse((o) => {
    if (!o.isLineSegments || !o.geometry || !o.geometry.attributes.position) return;
    const pos = o.geometry.attributes.position;
    const m = o.matrixWorld;
    const a = new THREE.Vector3(), b = new THREE.Vector3();
    for (let i = 0; i < pos.count; i += 2) {
      a.fromBufferAttribute(pos, i).applyMatrix4(m);
      b.fromBufferAttribute(pos, i + 1).applyMatrix4(m);
      out.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
  });
  return out;
}

function plotterPaths(E, cfg) {
  if (!E.wire) return null;
  const cam = E.camera;
  const aspect = cam.aspect || 1.5;
  const dw = Math.max(320, Math.min(2400, Math.round(cfg.quality)));
  const dh = Math.max(240, Math.round(dw / aspect));
  const depth = cfg.hlr ? depthPass(E, dw, dh) : null;

  cam.updateMatrixWorld(true);
  const viewProj = new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
  const ve = cam.matrixWorldInverse.elements;
  const edges = collectEdges(E.wire);
  const near = cam.near, far = cam.far, span = far - near;
  const tol = cfg.tolerance / 10000;
  const v4 = new THREE.Vector4();

  const depthAt = (nx, ny) => {
    const px = Math.min(dw - 1, Math.max(0, ((nx * 0.5 + 0.5) * dw) | 0));
    const py = Math.min(dh - 1, Math.max(0, ((ny * 0.5 + 0.5) * dh) | 0));
    const i = (py * dw + px) * 4;
    return depth[i] / 255 + depth[i + 1] / 65025;
  };

  const shown = [], hidden = [];
  let tested = 0;

  for (let e = 0; e < edges.length; e += 6) {
    const ax = edges[e], ay = edges[e + 1], az = edges[e + 2];
    const bx = edges[e + 3], by = edges[e + 4], bz = edges[e + 5];

    v4.set(ax, ay, az, 1).applyMatrix4(viewProj);
    if (v4.w <= 0) continue;
    const sax = v4.x / v4.w, say = v4.y / v4.w;
    v4.set(bx, by, bz, 1).applyMatrix4(viewProj);
    if (v4.w <= 0) continue;
    const sbx = v4.x / v4.w, sby = v4.y / v4.w;

    // sample about every 1.4 device pixels along the edge
    const pxLen = Math.hypot((sbx - sax) * dw * 0.5, (sby - say) * dh * 0.5);
    const steps = Math.max(1, Math.min(240, Math.ceil(pxLen / 1.4)));

    let run = null, hrun = null;
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const px = ax + (bx - ax) * t, py = ay + (by - ay) * t, pz = az + (bz - az) * t;
      v4.set(px, py, pz, 1).applyMatrix4(viewProj);
      if (v4.w <= 0) { run = null; hrun = null; continue; }
      const nx = v4.x / v4.w, ny = v4.y / v4.w;

      let visible = true;
      if (depth) {
        const vz = -(ve[2] * px + ve[6] * py + ve[10] * pz + ve[14]);
        visible = (vz - near) / span <= depthAt(nx, ny) + tol;
        tested++;
      }

      if (visible) {
        if (hrun) { if (hrun.length >= 4) hidden.push(hrun); hrun = null; }
        if (!run) run = [];
        run.push(nx, ny);
      } else {
        if (run) { if (run.length >= 4) shown.push(run); run = null; }
        if (cfg.hidden) { if (!hrun) hrun = []; hrun.push(nx, ny); }
      }
    }
    if (run && run.length >= 4) shown.push(run);
    if (hrun && hrun.length >= 4) hidden.push(hrun);
  }
  return { shown, hidden, tested, edgeCount: edges.length / 6 };
}

// Join paths whose ends touch. A plotter lifts the pen between paths, so
// thousands of disconnected fragments plot slowly and look ragged.
function chainPaths(paths, tol) {
  const key = (x, y) => Math.round(x / tol) + "," + Math.round(y / tol);
  const ends = new Map();
  const add = (k, i) => { if (!ends.has(k)) ends.set(k, []); ends.get(k).push(i); };
  paths.forEach((p, i) => {
    add(key(p[0], p[1]), i);
    add(key(p[p.length - 2], p[p.length - 1]), i);
  });
  const used = new Array(paths.length).fill(false);
  const out = [];
  for (let i = 0; i < paths.length; i++) {
    if (used[i]) continue;
    used[i] = true;
    const cur = paths[i].slice();
    let grew = true;
    while (grew) {
      grew = false;
      const k = key(cur[cur.length - 2], cur[cur.length - 1]);
      for (const j of ends.get(k) || []) {
        if (used[j]) continue;
        const q = paths[j];
        let seg = q;
        if (key(q[0], q[1]) !== k) {
          seg = [];
          for (let t = q.length - 2; t >= 0; t -= 2) seg.push(q[t], q[t + 1]);
        }
        for (let t = 2; t < seg.length; t += 2) cur.push(seg[t], seg[t + 1]);
        used[j] = true;
        grew = true;
        break;
      }
    }
    out.push(cur);
  }
  return out;
}

function pathLength(p) {
  let d = 0;
  for (let i = 2; i < p.length; i += 2) d += Math.hypot(p[i] - p[i - 2], p[i + 1] - p[i - 1]);
  return d;
}

// NDC to millimetres on the page, aspect preserved inside the margins.
function plotterSVG(paths, cfg, aspect) {
  const [pw, ph] = PAPERS[cfg.paper] || PAPERS.A4;
  const W = cfg.landscape ? ph : pw;
  const H = cfg.landscape ? pw : ph;
  const m = Math.max(0, Math.min(Math.min(W, H) / 2 - 5, cfg.margin));
  const availW = W - m * 2, availH = H - m * 2;

  const all = paths.shown.concat(paths.hidden);
  if (!all.length) return null;

  // Fit the drawing itself to the page, not the camera viewport. The model
  // rarely fills the frame, and scaling by the viewport wastes most of the paper.
  let x0 = 1, y0 = 1, x1 = -1, y1 = -1;
  if (cfg.fit) {
    for (const run of all)
      for (let i = 0; i < run.length; i += 2) {
        if (run[i] < x0) x0 = run[i];
        if (run[i] > x1) x1 = run[i];
        if (run[i + 1] < y0) y0 = run[i + 1];
        if (run[i + 1] > y1) y1 = run[i + 1];
      }
    if (x1 <= x0 || y1 <= y0) { x0 = -1; y0 = -1; x1 = 1; y1 = 1; }
  } else { x0 = -1; y0 = -1; x1 = 1; y1 = 1; }

  const spanX = (x1 - x0) / 2, spanY = (y1 - y0) / 2;
  const contentAspect = (spanX * aspect) / spanY;
  const drawW = Math.min(availW, availH * contentAspect);
  const drawH = drawW / contentAspect;
  const ox = (W - drawW) / 2, oy = (H - drawH) / 2;

  const toPage = (runs) =>
    runs.map((run) => {
      const out = new Array(run.length);
      for (let i = 0; i < run.length; i += 2) {
        out[i] = ox + ((run[i] - x0) / (x1 - x0)) * drawW;
        out[i + 1] = oy + (1 - (run[i + 1] - y0) / (y1 - y0)) * drawH;
      }
      return out;
    });

  const prepare = (runs) => {
    let out = toPage(runs);
    if (cfg.join) out = chainPaths(out, Math.max(0.02, cfg.pen * 0.5));
    if (cfg.minLen > 0) out = out.filter((p) => pathLength(p) >= cfg.minLen);
    return out;
  };

  const shown = prepare(paths.shown);
  // Occluded geometry is most of the model, so a reference layer has to be far
  // more aggressively culled than the drawing itself or it swamps it.
  const hidden = cfg.hidden
    ? prepare(paths.hidden).filter((p) => pathLength(p) >= Math.max(cfg.minLen, 2.5))
    : [];

  const poly = (run) => {
    let d = "";
    for (let i = 0; i < run.length; i += 2)
      d += (i ? "L" : "M") + run[i].toFixed(3) + " " + run[i + 1].toFixed(3);
    return `<path d="${d}"/>`;
  };
  const layer = (name, runs, colour, dashed) =>
    !runs.length ? "" :
    `<g inkscape:groupmode="layer" inkscape:label="${name}" stroke="${colour}" ` +
    `stroke-width="${(dashed ? cfg.pen * 0.6 : cfg.pen).toFixed(3)}" fill="none" ` +
    `stroke-linecap="round" stroke-linejoin="round"` +
    (dashed ? ` stroke-dasharray="${(cfg.pen * 4).toFixed(2)} ${(cfg.pen * 4).toFixed(2)}"` : "") +
    `>` + runs.map(poly).join("") + `</g>`;

  const ink = shown.concat(hidden).reduce((a2, p2) => a2 + pathLength(p2), 0);
  const lifts = shown.length + hidden.length;
  // rough AxiDraw figures: 50 mm/s drawing, 150 mm/s travelling, 0.18 s per lift
  const minutes = (ink / 50 + (lifts * 25) / 150 + lifts * 0.18) / 60;

  const svg =
    `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" ` +
    `width="${W}mm" height="${H}mm" viewBox="0 0 ${W} ${H}">` +
    layer("visible", shown, "#000000", false) +
    layer("hidden", hidden, "#cc2222", true) +
    `</svg>`;

  return { svg, paths: lifts, ink, minutes, W, H };
}


const SRC_DEFAULTS = { kind: "model", fit: "cover", mirror: true, cut: 0.06, name: "" };

const BUILD = "v44";
const MONO = '"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

/* ============================================================
   Component
   ============================================================ */

export default function Glyphworks() {
  const [s, setS] = useState(DEFAULTS);
  const [section, setSection] = useState("ascii");
  const [pc, setPc] = useState(PC_DEFAULTS);
  const [vx, setVx] = useState(VX_DEFAULTS);
  const [wf, setWf] = useState(WF_DEFAULTS);
  const [dt, setDt] = useState(DT_DEFAULTS);
  const [tl, setTl] = useState({ duration: 10, loop: true, ease: "smooth", seamless: true, keys: [] });
  const [playing, setPlaying] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [seq, setSeq] = useState({ busy: false, done: 0, total: 0 });
  const [exp, setExp] = useState({ height: 1080, fps: 30, field: "scene" });
  const [src, setSrc] = useState(SRC_DEFAULTS);
  const [hc, setHc] = useState(HC_DEFAULTS);
  const [plot, setPlot] = useState(PLOT_DEFAULTS);
  const [plotInfo, setPlotInfo] = useState(null);
  const [handSeen, setHandSeen] = useState(false);
  const [handStatus, setHandStatus] = useState("idle");
  const [md, setMd] = useState(MD_DEFAULTS);
  const [vid, setVid] = useState({ has: false, playing: false, dur: 0, loop: true });
  const [cl, setCl] = useState(CL_DEFAULTS);
  const [shoot, setShoot] = useState({ busy: false, count: 0, have: 0 });
  const [cloudInfo, setCloudInfo] = useState({ points: 0, voxels: 0, segs: 0 });
  const [modelName, setModelName] = useState("torus knot");
  const [stats, setStats] = useState({ cols: 0, rows: 0, tris: 0, fps: 0 });
  const [error, setErrorRaw] = useState("");
  const errorTimer = useRef(null);
  // Clear itself after a while: a stale banner over the picture reads as a
  // broken app long after whatever caused it has passed.
  const setError = useCallback((msg) => {
    setErrorRaw(msg);
    if (errorTimer.current) clearTimeout(errorTimer.current);
    if (msg) errorTimer.current = setTimeout(() => setErrorRaw(""), 12000);
  }, []);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [recording, setRecording] = useState(false);
  const [toast, setToast] = useState("");
  const [panelOpen, setPanelOpen] = useState(true);

  const canvasRef = useRef(null);
  const viewRef = useRef(null);
  const glRef = useRef(null);
  const pcRef = useRef(pc);
  const vxRef = useRef(vx);
  const wfRef = useRef(wf);
  const dtRef = useRef(dt);
  const tlRef = useRef(tl);
  const expRef = useRef(exp);
  const srcRef = useRef(src);
  const hcRef = useRef(hc);
  const handCvRef = useRef(null);
  const mdRef = useRef(md);
  const clRef = useRef(cl);
  const shotsRef = useRef([]);
  const vidHeadRef = useRef(null);
  const vidClockRef = useRef(null);
  const mediaInputRef = useRef(null);
  const imgInputRef = useRef(null);
  const play = useRef({ on: false, t: 0, baseYaw: 0 });
  const trackRef = useRef(null);
  const headRef = useRef(null);
  const clockRef = useRef(null);
  const capRef = useRef(null);
  const sectionRef = useRef(section);
  const fileRef = useRef(null);
  const engine = useRef(null);
  const sRef = useRef(s);
  const orbit = useRef({ yaw: 0.6, pitch: 0.25, dragging: false, lx: 0, ly: 0 });
  const textRef = useRef("");
  const triRef = useRef(0);
  const lastStats = useRef({ cols: 0, rows: 0, tris: 0 });
  const dotsRef = useRef(null);

  // While the timeline drives playback it owns these refs. Re-seeding them from
  // React state on an unrelated re-render throws one stale frame at the screen —
  // which is exactly the flicker that showed up in recordings.
  if (!play.current.on) {
    sRef.current = s;
    pcRef.current = pc;
    vxRef.current = vx;
    wfRef.current = wf;
    dtRef.current = dt;
  }
  tlRef.current = tl;
  expRef.current = exp;
  srcRef.current = src;
  hcRef.current = hc;
  mdRef.current = md;
  clRef.current = cl;

  // Declared up here because effects further down list these in their
  // dependency arrays — a const referenced before its declaration is a
  // temporal-dead-zone ReferenceError, not a hoisted undefined.
  // A live camera has no frame N until it happens, so timeline scrubbing and
  // frame-stepped export can't apply to it.
  const liveSource = src.kind === "camera";
  const mediaOn = src.kind !== "model" && (section === "ascii" || section === "dither" || section === "points");
  const mediaPts = src.kind !== "model" && section === "points";
  sectionRef.current = section;
  const set = (k, v) => setS((p) => ({ ...p, [k]: v }));
  const setP = (k, v) => setPc((p) => ({ ...p, [k]: v }));
  const setV = (k, v) => setVx((p) => ({ ...p, [k]: v }));
  const setW = (k, v) => setWf((p) => ({ ...p, [k]: v }));
  const setD = (k, v) => setDt((p) => ({ ...p, [k]: v }));
  const setM = (k, v) => setMd((p) => ({ ...p, [k]: v }));
  const setH = (k, v) => setHc((p) => ({ ...p, [k]: v }));
  const setPl = (k, v) => setPlot((p) => ({ ...p, [k]: v }));
  const setC = (k, v) => setCl((p) => ({ ...p, [k]: v }));
  const refFor = (sec) =>
    sec === "ascii" ? sRef : sec === "dither" ? dtRef : sec === "points" ? pcRef : sec === "voxel" ? vxRef : wfRef;
  const setterFor = (sec) =>
    sec === "ascii" ? setS : sec === "dither" ? setDt : sec === "points" ? setPc : sec === "voxel" ? setVx : setWf;
  const cfgFor = (sec) =>
    sec === "points" ? pcRef.current : sec === "voxel" ? vxRef.current : sec === "wire" ? wfRef.current : dtRef.current;

  const flash = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(""), 1800);
  };

  /* ---- engine boot ---- */
  useEffect(() => {
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setClearColor(0x000000, 0);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 200);
    const pivot = new THREE.Group();
    scene.add(pivot);

    const ambient = new THREE.AmbientLight(0xffffff, 0.2);
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    const fill = new THREE.DirectionalLight(0xffffff, 0.4);
    fill.position.set(-3, -1, 2);
    scene.add(ambient, key, fill);

    const depthMat = new THREE.MeshDepthMaterial();
    const normalMat = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });

    if (glRef.current) glRef.current.appendChild(renderer.domElement);
    engine.current = { renderer, scene, camera, pivot, ambient, key, fill, depthMat, normalMat,
      cap: 0, capApplied: 0, bgOver: null, capturing: false, src: null, scv: null,
      mpts: null, mjit: null, mcv: null, mgrid: "",
      rt: null, rtC: null, buf: null, bufC: null, content: null,
      points: null, voxels: null, wire: null, disc: discTexture(), vw: 0, vh: 0 };

    const first = primitive("knot");
    pivot.add(first.group);
    engine.current.content = first.group;
    triRef.current = first.tris;

    let raf, last = performance.now(), frames = 0, acc = 0, lastLive = false;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const now = performance.now();
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      acc += dt;
      frames++;
      if (acc > 0.5) {
        const fps = Math.round(frames / acc);
        frames = 0;
        acc = 0;
        if (!play.current.on) setStats((p) => ({ ...p, fps }));
      }
      if (engine.current && engine.current.capturing) return;
      const sec = sectionRef.current;
      const hcCfg = hcRef.current;
      if (hcCfg.on && !engine.current.capturing) {
        const st = readHand(engine.current, hcCfg);
        if (st) {
          applyHand(st, hcCfg, orbit.current, refFor(sec));
          if (st.seen !== lastLive) { lastLive = st.seen; setHandSeen(st.seen); }
          const pv = handCvRef.current;
          const E2 = engine.current;
          if (pv && E2.ctrl && E2.ctrl.el) {
            const g = pv.getContext("2d");
            const iw = 120, ih = 90;
            if (pv.width !== iw) { pv.width = iw; pv.height = ih; }
            g.setTransform(1, 0, 0, 1, 0, 0);
            g.clearRect(0, 0, iw, ih);
            g.save();
            if (hcCfg.mirror) { g.translate(iw, 0); g.scale(-1, 1); }
            try { g.drawImage(E2.ctrl.el, 0, 0, iw, ih); } catch {}
            g.restore();
            if (st.seen && st.pts) {
              const px = (p) => (hcCfg.mirror ? (1 - p.x) : p.x) * iw;
              const py = (p) => p.y * ih;
              g.strokeStyle = "rgba(120,255,180,0.9)";
              g.lineWidth = 1.4;
              g.beginPath();
              for (const [a2, b2] of BONES) {
                g.moveTo(px(st.pts[a2]), py(st.pts[a2]));
                g.lineTo(px(st.pts[b2]), py(st.pts[b2]));
              }
              g.stroke();
              g.fillStyle = "rgba(91,140,255,0.95)";
              for (const p of st.pts) { g.beginPath(); g.arc(px(p), py(p), 1.6, 0, Math.PI * 2); g.fill(); }
            }
          }
        }
      }
      if (play.current.on) {
        const D = tlRef.current.duration;
        play.current.t += dt;
        if (play.current.t >= D) {
          if (tlRef.current.loop) play.current.t -= D;
          else { play.current.t = D; play.current.on = false; setPlaying(false); }
        }
        applyTime(play.current.t);
        if (headRef.current) headRef.current.style.left = (play.current.t / D) * 100 + "%";
        if (clockRef.current) clockRef.current.textContent = play.current.t.toFixed(2) + "s";
      } else {
        const spin = sec === "ascii" ? sRef.current.spin : cfgFor(sec).spin;
        if (!orbit.current.dragging) orbit.current.yaw += spin * dt;
      }
      if (sec === "collage") drawCollage();
      else if (sec === "media") drawMedia();
      else if (sec === "ascii") draw(dt);
      else if (sec === "dither") drawDither();
      else draw3D(cfgFor(sec), sec);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      renderer.dispose();
    };
    // eslint-disable-next-line
  }, []);

  /* ---- the frame ---- */
  const draw = useCallback(() => {
    const E = engine.current;
    const cv = canvasRef.current;
    if (!E || !cv) return;
    const cfg = sRef.current;

    const host = viewRef.current;
    const W = host ? host.clientWidth : cv.clientWidth;
    const H = host ? host.clientHeight : cv.clientHeight;
    if (W < 8 || H < 8) return;
    const dpr = E.cap || Math.min(window.devicePixelRatio || 1, 2);
    let cw = Math.floor(W * dpr), ch = Math.floor(H * dpr);
    if (E.cap) { cw -= cw % 2; ch -= ch % 2; }
    if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const cols = Math.max(16, Math.round(cfg.detail));
    const cellW = W / cols;
    ctx.font = `100px ${MONO}`;
    const adv = ctx.measureText("M").width / 100 || 0.6;
    const fontSize = cellW / adv;
    const cellH = (cfg.mark === "dots" ? cellW : fontSize) * cfg.lineHeight;
    const rows = Math.max(6, Math.floor(H / cellH));
    const yPad = (H - rows * cellH) / 2;

    // render target
    if (!E.rt || E.rt.width !== cols || E.rt.height !== rows) {
      if (E.rt) E.rt.dispose();
      E.rt = new THREE.WebGLRenderTarget(cols, rows, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat,
      });
      if ("SRGBColorSpace" in THREE) E.rt.texture.colorSpace = THREE.SRGBColorSpace;
      E.buf = new Uint8Array(cols * rows * 4);
      E.renderer.setSize(cols, rows, false);
    }

    // scene state
    E.pivot.rotation.set(orbit.current.pitch, orbit.current.yaw, 0);
    E.pivot.scale.setScalar(cfg.scale);
    E.pivot.position.y = cfg.height;
    E.camera.aspect = W / (rows * cellH);
    E.camera.position.set(0, 0, cfg.zoom);
    E.camera.lookAt(0, 0, 0);
    E.camera.updateProjectionMatrix();
    E.ambient.intensity = cfg.ambient;
    E.key.intensity = cfg.key;
    E.fill.intensity = cfg.fill;
    const a = (cfg.keyAngle * Math.PI) / 180;
    E.key.position.set(Math.cos(a) * 4, 2.5, Math.sin(a) * 4);
    E.scene.overrideMaterial =
      cfg.pass === "depth" ? E.depthMat : cfg.pass === "normal" ? E.normalMat : null;

    const media = srcRef.current.kind !== "model" && E.src && E.src.el;
    let gotMedia = false;
    if (media) gotMedia = fillFromSource(E, cols, rows, cellW, cellH, srcRef.current);
    if (!gotMedia) {
      E.renderer.setRenderTarget(E.rt);
      E.renderer.clear();
      E.renderer.render(E.scene, E.camera);
      E.renderer.setRenderTarget(null);
      E.renderer.readRenderTargetPixels(E.rt, 0, 0, cols, rows, E.buf);
    }

    // Colour comes from an unlit pass so highlights can't bleach the hue.
    const wantFlat = !gotMedia && cfg.color === "model" && cfg.flat && E.content;
    if (wantFlat) {
      if (!E.rtC || E.rtC.width !== cols || E.rtC.height !== rows) {
        if (E.rtC) E.rtC.dispose();
        E.rtC = new THREE.WebGLRenderTarget(cols, rows, {
          minFilter: THREE.NearestFilter,
          magFilter: THREE.NearestFilter,
          format: THREE.RGBAFormat,
        });
        if ("SRGBColorSpace" in THREE) E.rtC.texture.colorSpace = THREE.SRGBColorSpace;
        E.bufC = new Uint8Array(cols * rows * 4);
      }
      swapMaterials(E.content, "flat");
      E.scene.overrideMaterial = null;
      E.renderer.setRenderTarget(E.rtC);
      E.renderer.clear();
      E.renderer.render(E.scene, E.camera);
      E.renderer.setRenderTarget(null);
      E.renderer.readRenderTargetPixels(E.rtC, 0, 0, cols, rows, E.bufC);
      swapMaterials(E.content, "shaded");
    }

    // luminance grid (rows flipped; empty space stays empty)
    const n = cols * rows;
    const lum = new Float32Array(n);
    const hit = new Uint8Array(n);
    const buf = E.buf;
    let lo = 1, hi = 0;
    for (let r = 0; r < rows; r++) {
      const src = (rows - 1 - r) * cols * 4;
      for (let c = 0; c < cols; c++) {
        const i = src + c * 4;
        const k = r * cols + c;
        if (buf[i + 3] < 8) continue;
        hit[k] = 1;
        const v = (0.299 * buf[i] + 0.587 * buf[i + 1] + 0.114 * buf[i + 2]) / 255;
        lum[k] = v;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    // depth sits in a narrow band, so stretch it back out
    const span = !gotMedia && cfg.pass === "depth" && hi - lo > 0.001 ? 1 / (hi - lo) : 0;
    for (let k = 0; k < n; k++) {
      if (!hit[k]) { lum[k] = 0; continue; }
      let v = span ? (lum[k] - lo) * span : lum[k];
      v = (v - 0.5) * cfg.contrast + 0.5 + cfg.brightness;
      if (cfg.gamma !== 1) v = Math.pow(Math.max(v, 0), cfg.gamma);
      if (cfg.invert) v = 1 - v;
      lum[k] = Math.min(1, Math.max(0, v));
    }

    const cbuf = wantFlat && E.bufC ? E.bufC : buf;
    const sat = cfg.sat;
    const rgbAt = (i) => {
      const R = cbuf[i], G = cbuf[i + 1], B = cbuf[i + 2];
      if (sat === 1) return `rgb(${R},${G},${B})`;
      const m = (R + G + B) / 3;
      return `rgb(${sat255(m + (R - m) * sat)},${sat255(m + (G - m) * sat)},${sat255(m + (B - m) * sat)})`;
    };

    const ramp = (cfg.ramp === "custom" ? cfg.custom : RAMPS[cfg.ramp]) || RAMPS.standard;
    const last = ramp.length - 1;

    // glyph grid
    const grid = new Array(rows);
    for (let r = 0; r < rows; r++) {
      let line = "";
      for (let c = 0; c < cols; c++) {
        const k = r * cols + c;
        const v = lum[k];
        let ch = hit[k] ? ramp[Math.round(v * last)] || " " : " ";
        if (cfg.edges && cfg.mark === "glyphs" && r > 0 && r < rows - 1 && c > 0 && c < cols - 1) {
          const g = (dr, dc) => lum[(r + dr) * cols + (c + dc)];
          const gx =
            -g(-1, -1) - 2 * g(0, -1) - g(1, -1) + g(-1, 1) + 2 * g(0, 1) + g(1, 1);
          const gy =
            -g(-1, -1) - 2 * g(-1, 0) - g(-1, 1) + g(1, -1) + 2 * g(1, 0) + g(1, 1);
          const mag = Math.hypot(gx, gy);
          if (mag > cfg.edgeAmount * 4) {
            const ang = ((Math.atan2(gy, gx) + Math.PI) / Math.PI) * 2;
            ch = EDGE_GLYPHS[Math.round(ang) % 4];
          }
        }
        line += ch;
      }
      grid[r] = line;
    }
    textRef.current = grid.join("\n");

    if (cfg.mark === "dots") {
      const colors = new Array(n);
      for (let r = 0; r < rows; r++) {
        const src = (rows - 1 - r) * cols * 4;
        for (let c = 0; c < cols; c++) {
          const i = src + c * 4;
          colors[r * cols + c] = rgbAt(i);
        }
      }
      dotsRef.current = { cols, rows, lum, hit, colors };
    }

    // paint
    if (E.alpha) ctx.clearRect(0, 0, W, H);
    else { ctx.fillStyle = E.bgOver || cfg.bg; ctx.fillRect(0, 0, W, H); }
    ctx.font = `${fontSize.toFixed(2)}px ${MONO}`;
    ctx.textBaseline = "top";
    ctx.textAlign = "left";

    if (cfg.glow) {
      ctx.shadowColor = cfg.tint;
      ctx.shadowBlur = fontSize * 0.9;
    } else {
      ctx.shadowBlur = 0;
    }

    if (cfg.mark === "dots") {
      const half = cellW / 2;
      const maxR = Math.min(cellW, cellH) * 0.5 * cfg.dot;
      const ring = cfg.shape === "ring";
      const square = cfg.shape === "square";
      const mono = cfg.color === "mono";
      const t = hexToRgb(cfg.tint);

      const plot = (cx, cy, rad) => {
        if (square) ctx.rect(cx - rad, cy - rad, rad * 2, rad * 2);
        else {
          ctx.moveTo(cx + rad, cy);
          ctx.arc(cx, cy, rad, 0, Math.PI * 2);
        }
      };

      if (mono) {
        ctx.beginPath();
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const k = r * cols + c;
            if (!hit[k]) continue;
            const rad = maxR * Math.sqrt(lum[k]);
            if (rad < 0.12) continue;
            plot(c * cellW + half, yPad + r * cellH + cellH / 2, rad);
          }
        }
        if (ring) {
          ctx.strokeStyle = cfg.tint;
          ctx.lineWidth = Math.max(0.6, maxR * 0.28);
          ctx.stroke();
        } else {
          ctx.fillStyle = cfg.tint;
          ctx.fill();
        }
      } else {
        ctx.lineWidth = Math.max(0.6, maxR * 0.28);
        for (let r = 0; r < rows; r++) {
          const src = (rows - 1 - r) * cols * 4;
          for (let c = 0; c < cols; c++) {
            const k = r * cols + c;
            if (!hit[k]) continue;
            const v = lum[k];
            const rad = maxR * Math.sqrt(v);
            if (rad < 0.12) continue;
            const i = src + c * 4;
            const col =
              cfg.color === "model"
                ? rgbAt(i)
                : `rgb(${Math.round(t.r * (0.35 + 0.65 * v))},${Math.round(
                    t.g * (0.2 + 0.8 * v)
                  )},${Math.round(t.b * (0.55 + 0.45 * v))})`;
            ctx.beginPath();
            plot(c * cellW + half, yPad + r * cellH + cellH / 2, rad);
            if (ring) {
              ctx.strokeStyle = col;
              ctx.stroke();
            } else {
              ctx.fillStyle = col;
              ctx.fill();
            }
          }
        }
      }
    } else if (cfg.color === "mono") {
      ctx.fillStyle = cfg.tint;
      for (let r = 0; r < rows; r++) ctx.fillText(grid[r], 0, yPad + r * cellH);
    } else {
      for (let r = 0; r < rows; r++) {
        const src = (rows - 1 - r) * cols * 4;
        for (let c = 0; c < cols; c++) {
          const ch = grid[r][c];
          if (ch === " ") continue;
          const i = src + c * 4;
          if (cfg.color === "model") {
            ctx.fillStyle = rgbAt(i);
          } else {
            const v = lum[r * cols + c];
            const t = hexToRgb(cfg.tint);
            ctx.fillStyle = `rgb(${Math.round(t.r * (0.35 + 0.65 * v))},${Math.round(
              t.g * (0.2 + 0.8 * v)
            )},${Math.round(t.b * (0.55 + 0.45 * v))})`;
          }
          ctx.fillText(ch, c * cellW, yPad + r * cellH);
        }
      }
    }
    ctx.shadowBlur = 0;

    if (cfg.scanlines) {
      ctx.fillStyle = E.bgOver || cfg.bg;
      ctx.globalAlpha = 0.35;
      for (let y = 0; y < H; y += 3) ctx.fillRect(0, y, W, 1);
      ctx.globalAlpha = 1;
    }

    const L = lastStats.current;
    if (L.cols !== cols || L.rows !== rows || L.tris !== triRef.current) {
      lastStats.current = { cols, rows, tris: triRef.current };
      setStats((p) => ({ ...p, cols, rows, tris: triRef.current }));
    }
  }, []);

  /* ---- timeline ---- */
  const sectionKeys = useCallback(
    (sec) => tlRef.current.keys.filter((k) => k.section === sec).sort((a, b) => a.t - b.t),
    []
  );

  // Writes interpolated values straight into the section's ref, bypassing
  // React entirely — a state update per frame would stall playback.
  const applyTime = useCallback((t) => {
    const sec = sectionRef.current;
    const keys = sectionKeys(sec);
    const cfgRef = refFor(sec);
    const spec = ANIM[sec];
    if (!keys.length) {
      // no keys: still drive rotation deterministically so exports repeat exactly
      const spin = sec === "ascii" ? sRef.current.spin : cfgRef.current.spin;
      orbit.current.yaw = play.current.baseYaw + spin * t;
      return;
    }
    let a = keys[0], b = keys[keys.length - 1];
    for (let i = 0; i < keys.length - 1; i++) {
      if (t >= keys[i].t && t <= keys[i + 1].t) { a = keys[i]; b = keys[i + 1]; break; }
    }
    if (t <= keys[0].t) { a = keys[0]; b = keys[0]; }
    if (t >= keys[keys.length - 1].t) { a = b = keys[keys.length - 1]; }
    const span = b.t - a.t;
    const f = span > 0.0001 ? (EASE[tlRef.current.ease] || EASE.smooth)((t - a.t) / span) : 0;

    orbit.current.yaw = a.snap.yaw + (b.snap.yaw - a.snap.yaw) * f;
    orbit.current.pitch = a.snap.pitch + (b.snap.pitch - a.snap.pitch) * f;

    const next = { ...cfgRef.current };
    for (const k of spec.num)
      if (a.snap[k] !== undefined && b.snap[k] !== undefined)
        next[k] = a.snap[k] + (b.snap[k] - a.snap[k]) * f;
    for (const k of spec.col)
      if (a.snap[k] !== undefined && b.snap[k] !== undefined) next[k] = lerpHex(a.snap[k], b.snap[k], f);
    cfgRef.current = next;
  }, [sectionKeys]);

  const addKey = useCallback(() => {
    const sec = sectionRef.current;
    const cfg = refFor(sec).current;
    const spec = ANIM[sec];
    const snap = { yaw: orbit.current.yaw, pitch: orbit.current.pitch };
    for (const k of spec.num) snap[k] = cfg[k];
    for (const k of spec.col) snap[k] = cfg[k];
    const t = Math.round(play.current.t * 100) / 100;
    setTl((p) => ({
      ...p,
      keys: [...p.keys.filter((k) => !(k.section === sec && Math.abs(k.t - t) < 0.02)),
             { id: Date.now() + Math.random(), t, section: sec, snap }],
    }));
    flash("Key set at " + t.toFixed(2) + "s");
  }, []);

  const seekTo = useCallback((t) => {
    play.current.t = Math.max(0, Math.min(tlRef.current.duration, t));
    applyTime(play.current.t);
    if (headRef.current) headRef.current.style.left = (play.current.t / tlRef.current.duration) * 100 + "%";
    if (clockRef.current) clockRef.current.textContent = play.current.t.toFixed(2) + "s";
  }, [applyTime]);

  // pull the animated values back into React state when playback stops
  const syncBack = useCallback(() => {
    const sec = sectionRef.current;
    const cfgRef = refFor(sec);
    setterFor(sec)({ ...cfgRef.current });
  }, []);

  const snapSpin = useCallback((D) => {
    const sec = sectionRef.current;
    const cfgRef = refFor(sec);
    const spin = cfgRef.current.spin;
    if (!spin) return;
    const turns = Math.max(1, Math.round((Math.abs(spin) * D) / (Math.PI * 2)));
    const exact = Math.sign(spin) * ((turns * Math.PI * 2) / D);
    cfgRef.current = { ...cfgRef.current, spin: exact };
  }, []);

  /* ---- dithering ---- */
  const drawDither = useCallback(() => {
    const E = engine.current, cv = canvasRef.current, host = viewRef.current;
    if (!E || !cv || !host) return;
    const cfg = dtRef.current;
    const W = host.clientWidth, H = host.clientHeight;
    if (W < 8 || H < 8) return;
    const dpr = E.cap || Math.min(window.devicePixelRatio || 1, 2);
    let cw = Math.floor(W * dpr), ch = Math.floor(H * dpr);
    if (E.cap) { cw -= cw % 2; ch -= ch % 2; } // encoders reject odd dimensions
    if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const cols = Math.max(24, Math.round(cfg.res));
    const cell = W / cols;
    const rows = Math.max(8, Math.floor(H / cell));
    const yPad = (H - rows * cell) / 2;

    if (!E.rt || E.rt.width !== cols || E.rt.height !== rows) {
      if (E.rt) E.rt.dispose();
      E.rt = new THREE.WebGLRenderTarget(cols, rows, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat,
      });
      if ("SRGBColorSpace" in THREE) E.rt.texture.colorSpace = THREE.SRGBColorSpace;
      E.buf = new Uint8Array(cols * rows * 4);
      E.renderer.setSize(cols, rows, false);
      E.vw = 0;
    }

    E.pivot.rotation.set(orbit.current.pitch, orbit.current.yaw, 0);
    E.pivot.scale.setScalar(cfg.scale);
    E.pivot.position.y = cfg.height;
    E.camera.aspect = W / (rows * cell);
    E.camera.position.set(0, 0, cfg.zoom);
    E.camera.lookAt(0, 0, 0);
    E.camera.updateProjectionMatrix();
    E.ambient.intensity = cfg.ambient;
    E.key.intensity = cfg.key;
    E.fill.intensity = 0.35;
    E.key.position.set(3, 3, 3);
    E.scene.fog = null;
    E.scene.overrideMaterial = null;
    const media = srcRef.current.kind !== "model" && E.src && E.src.el;
    if (!media || !fillFromSource(E, cols, rows, cell, cell, srcRef.current)) {
      E.renderer.setRenderTarget(E.rt);
      E.renderer.clear();
      E.renderer.render(E.scene, E.camera);
      E.renderer.setRenderTarget(null);
      E.renderer.readRenderTargetPixels(E.rt, 0, 0, cols, rows, E.buf);
    }

    const n = cols * rows;
    const buf = E.buf;
    const work = new Float32Array(n * 3);
    const mask = new Uint8Array(n);
    const bgRGB = hexList([E.bgOver || cfg.bg])[0];

    for (let r = 0; r < rows; r++) {
      const src = (rows - 1 - r) * cols * 4;
      for (let c = 0; c < cols; c++) {
        const i = src + c * 4;
        const k = r * cols + c;
        if (buf[i + 3] < 8) continue;
        mask[k] = 1;
        for (let ch = 0; ch < 3; ch++) {
          let v = buf[i + ch] / 255;
          v = (v - 0.5) * cfg.contrast + 0.5 + cfg.brightness;
          if (cfg.gamma !== 1) v = Math.pow(Math.max(v, 0), cfg.gamma);
          if (cfg.invert) v = 1 - v;
          work[k * 3 + ch] = Math.min(1, Math.max(0, v)) * 255;
        }
      }
    }

    const usePost = cfg.palette === "posterise";
    const pal = usePost
      ? null
      : cfg.palette === "mono"
      ? hexList([cfg.bg, cfg.ink])
      : hexList(PALETTES[cfg.palette]);
    const L = Math.max(2, Math.round(cfg.levels));
    const quant = (r, g, b, out) => {
      if (usePost) {
        out[0] = Math.round((r / 255) * (L - 1)) * (255 / (L - 1));
        out[1] = Math.round((g / 255) * (L - 1)) * (255 / (L - 1));
        out[2] = Math.round((b / 255) * (L - 1)) * (255 / (L - 1));
      } else {
        const p = nearestIn(pal, r, g, b);
        out[0] = p[0]; out[1] = p[1]; out[2] = p[2];
      }
    };

    const out = [0, 0, 0];
    if (cfg.algo === "fs") {
      // serpentine scan keeps the error from drifting in one direction
      for (let r = 0; r < rows; r++) {
        const ltr = r % 2 === 0;
        for (let j = 0; j < cols; j++) {
          const c = ltr ? j : cols - 1 - j;
          const k = r * cols + c;
          if (!mask[k]) continue;
          const or = work[k * 3], og = work[k * 3 + 1], ob = work[k * 3 + 2];
          quant(or, og, ob, out);
          work[k * 3] = out[0]; work[k * 3 + 1] = out[1]; work[k * 3 + 2] = out[2];
          const er = (or - out[0]) * cfg.spread;
          const eg = (og - out[1]) * cfg.spread;
          const eb = (ob - out[2]) * cfg.spread;
          const push = (cc, rr, f) => {
            if (cc < 0 || cc >= cols || rr >= rows) return;
            const kk = rr * cols + cc;
            if (!mask[kk]) return;
            work[kk * 3] += er * f;
            work[kk * 3 + 1] += eg * f;
            work[kk * 3 + 2] += eb * f;
          };
          const d = ltr ? 1 : -1;
          push(c + d, r, 7 / 16);
          push(c - d, r + 1, 3 / 16);
          push(c, r + 1, 5 / 16);
          push(c + d, r + 1, 1 / 16);
        }
      }
    } else {
      const M = BAYER[cfg.bayer] || BAYER[4];
      const m = M.length;
      const scale = (255 / (usePost ? L : 2)) * cfg.spread;
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          const k = r * cols + c;
          if (!mask[k]) continue;
          const t = cfg.algo === "none" ? 0 : (M[r % m][c % m] / (m * m) - 0.5) * scale;
          quant(work[k * 3] + t, work[k * 3 + 1] + t, work[k * 3 + 2] + t, out);
          work[k * 3] = out[0]; work[k * 3 + 1] = out[1]; work[k * 3 + 2] = out[2];
        }
    }

    if (!E.dcv || E.dcv.width !== cols || E.dcv.height !== rows) {
      E.dcv = document.createElement("canvas");
      E.dcv.width = cols;
      E.dcv.height = rows;
      E.dctx = E.dcv.getContext("2d");
      E.dimg = E.dctx.createImageData(cols, rows);
    }
    const px = E.dimg.data;
    for (let k = 0; k < n; k++) {
      const o = k * 4;
      if (mask[k]) {
        px[o] = work[k * 3];
        px[o + 1] = work[k * 3 + 1];
        px[o + 2] = work[k * 3 + 2];
        px[o + 3] = 255;
      } else {
        px[o] = bgRGB[0]; px[o + 1] = bgRGB[1]; px[o + 2] = bgRGB[2];
        px[o + 3] = E.alpha ? 0 : 255;
      }
    }
    E.dctx.putImageData(E.dimg, 0, 0);

    if (E.alpha) ctx.clearRect(0, 0, W, H);
    else { ctx.fillStyle = E.bgOver || cfg.bg; ctx.fillRect(0, 0, W, H); }
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(E.dcv, 0, yPad, W, rows * cell);

    const S = lastStats.current;
    if (S.cols !== cols || S.rows !== rows) {
      lastStats.current = { ...S, cols, rows };
      setStats((p) => ({ ...p, cols, rows }));
    }
  }, []);

  /* ---- point cloud ---- */
  const draw3D = useCallback((cfg, kind) => {
    const E = engine.current, host = viewRef.current;
    if (!E || !host || !cfg) return;
    const W = host.clientWidth, H = host.clientHeight;
    if (W < 8 || H < 8) return;
    if (E.vw !== W || E.vh !== H || E.capApplied !== E.cap) {
      E.capApplied = E.cap;
      if (E.cap) {
        let pw = Math.floor(W * E.cap), ph = Math.floor(H * E.cap);
        pw -= pw % 2; ph -= ph % 2;
        E.renderer.setPixelRatio(1);
        E.renderer.setSize(pw, ph, false);
      } else {
        E.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        E.renderer.setSize(W, H, false);
      }
      E.vw = W; E.vh = H;
    }
    E.renderer.setClearColor(new THREE.Color(E.bgOver || cfg.bg), E.alpha ? 0 : 1);
    if (kind === "points" && srcRef.current.kind !== "model" && E.src && E.src.el) {
      const k = updateMediaPoints(E, cfg, srcRef.current);
      const m = E.mpts;
      if (m) {
        m.visible = true;
        m.material.size = cfg.size;
        m.material.opacity = cfg.opacity;
        m.material.map = cfg.shape === "round" ? E.disc : null;
        m.material.alphaTest = cfg.shape === "round" ? 0.06 : 0;
        m.material.blending = cfg.additive ? THREE.AdditiveBlending : THREE.NormalBlending;
        m.material.depthWrite = !cfg.additive;
        m.material.sizeAttenuation = cfg.atten;
        const sh = m.material.userData.sh;
        if (sh && sh.uniforms.uMinPx)
          sh.uniforms.uMinPx.value = Math.max(1, (E.renderer.domElement.height / 400) * cfg.minPx);
      }
      if (E.points) E.points.visible = false;
      if (!E.capturing && Math.abs(k - (E.mlast || 0)) > Math.max(64, k * 0.03)) {
        E.mlast = k;
        setCloudInfo((p) => ({ ...p, points: k }));
      }
    } else if (kind === "points") {
      if (E.mpts) E.mpts.visible = false;
      if (E.points) E.points.visible = true;
    }
    if (kind === "points" && E.points && E.points.visible && E.pScatter !== cfg.scatter) {
      const g = E.points.geometry, base = g.userData.base, dir = g.userData.dir;
      const arr = g.attributes.position.array;
      for (let i = 0; i < arr.length; i++) arr[i] = base[i] + dir[i] * cfg.scatter;
      g.attributes.position.needsUpdate = true;
      E.pScatter = cfg.scatter;
    }
    if (kind === "points" && E.points && E.points.visible) {
      const m = E.points.material;
      m.size = cfg.size;
      m.opacity = cfg.opacity;
      const sh = m.userData.sh;
      if (sh && sh.uniforms.uMinPx)
        sh.uniforms.uMinPx.value = Math.max(1, (E.renderer.domElement.height / 400) * cfg.minPx);
    }
    if (kind === "voxel" && E.voxels && E.vScatter !== cfg.scatter) {
      placeVoxels(E.voxels, cfg.scatter);
      E.vScatter = cfg.scatter;
    }
    if (kind === "wire" && E.wire && E.wire.userData.lineMat) {
      const lm = E.wire.userData.lineMat;
      lm.opacity = cfg.opacity;
      lm.color.set(cfg.color);
    }
    if (kind === "voxel") {
      E.ambient.intensity = cfg.lit ? cfg.ambient : 1;
      E.key.intensity = cfg.lit ? cfg.key : 0;
      E.fill.intensity = cfg.lit ? 0.35 : 0;
      E.key.position.set(3, 3.5, 2.5);
    }
    if (kind === "wire" && E.half) {
      const r = Math.hypot(E.half.x, E.half.y, E.half.z) * cfg.scale;
      if (cfg.fog > 0.01) {
        if (!E.scene.fog) E.scene.fog = new THREE.Fog(0x000000, 1, 10);
        E.scene.fog.color.set(cfg.bg);
        E.scene.fog.near = Math.max(0.01, cfg.zoom - r);
        E.scene.fog.far = cfg.zoom + r * (0.3 + (1 - cfg.fog) * 4);
      } else E.scene.fog = null;
    } else E.scene.fog = null;
    E.pivot.rotation.set(orbit.current.pitch, orbit.current.yaw, 0);
    E.pivot.scale.setScalar(cfg.scale);
    E.pivot.position.y = cfg.height;
    E.camera.aspect = W / H;
    E.camera.position.set(0, 0, cfg.zoom);
    E.camera.lookAt(0, 0, 0);
    E.camera.updateProjectionMatrix();
    E.scene.overrideMaterial = null;
    E.renderer.setRenderTarget(null);
    E.renderer.render(E.scene, E.camera);
  }, []);

  const rebuildCloud = useCallback(() => {
    const E = engine.current;
    if (!E || !E.content) return;
    const cfg = pcRef.current;
    const data = samplePoints(E.content, cfg);
    if (!data) return;
    if (E.points) {
      E.pivot.remove(E.points);
      E.points.geometry.dispose();
      E.points.material.dispose();
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(data.position, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(data.color, 3));
    geo.userData.base = data.position.slice();
    geo.userData.dir = data.dir;
    const mat = new THREE.PointsMaterial({
      size: cfg.size,
      sizeAttenuation: cfg.atten,
      vertexColors: true,
      transparent: true,
      opacity: cfg.opacity,
      map: cfg.shape === "round" ? E.disc : null,
      alphaTest: cfg.shape === "round" ? 0.06 : 0,
      depthWrite: !cfg.additive,
      blending: cfg.additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    });
    // Enforce a floor on gl_PointSize. Sub-pixel sprites alias violently as
    // they rotate, which is what reads as flicker in an exported clip.
    mat.onBeforeCompile = (sh) => {
      sh.uniforms.uMinPx = { value: 2.5 };
      sh.vertexShader = "uniform float uMinPx;\n" + sh.vertexShader;
      sh.vertexShader = sh.vertexShader.replace(
        "#include <fog_vertex>",
        "gl_PointSize = max( gl_PointSize, uMinPx );\n\t#include <fog_vertex>"
      );
      mat.userData.sh = sh;
    };
    E.points = new THREE.Points(geo, mat);
    E.pivot.add(E.points);
    setCloudInfo({ points: data.count });
  }, []);

  const rebuildVoxels = useCallback(() => {
    const E = engine.current;
    if (!E || !E.content) return;
    const cfg = vxRef.current;
    const data = buildVoxels(E.content, cfg);
    if (!data) return;
    if (E.voxels) {
      E.pivot.remove(E.voxels);
      E.voxels.geometry.dispose();
      E.voxels.material.dispose();
    }
    const size = data.step * (1 - cfg.gap);
    const geo = new THREE.BoxGeometry(size, size, size);
    const mat = cfg.lit
      ? new THREE.MeshLambertMaterial({ color: 0xffffff })
      : new THREE.MeshBasicMaterial({ color: 0xffffff });
    const mesh = new THREE.InstancedMesh(geo, mat, data.list.length);
    const col = new THREE.Color();
    const tintable = typeof mesh.setColorAt === "function";
    if (tintable)
      data.list.forEach((c, i) => {
        col.setRGB(c.r / c.k, c.g / c.k, c.b / c.k);
        mesh.setColorAt(i, col);
      });
    mesh.userData.cells = data;
    placeVoxels(mesh, cfg.scatter);
    if (tintable && mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    E.voxels = mesh;
    E.pivot.add(mesh);
    setCloudInfo((p) => ({ ...p, voxels: data.list.length }));
  }, []);

  const rebuildWire = useCallback(() => {
    const E = engine.current;
    if (!E || !E.content) return;
    if (E.wire) {
      E.pivot.remove(E.wire);
      disposeWire(E.wire);
    }
    const g = buildWire(E.content, wfRef.current);
    E.wire = g;
    E.pivot.add(g);
    setCloudInfo((p) => ({ ...p, segs: g.userData.segs }));
  }, []);

  useEffect(() => {
    const E = engine.current;
    if (!E) return;
    if (E.content) E.content.visible = section === "ascii" || section === "dither";
    if (section === "media" || section === "collage") { E.vw = 0; E.scene.fog = null; }
    if (E.points) E.points.visible = section === "points" && !mediaPts;
    if (E.mpts) E.mpts.visible = section === "points" && mediaPts;
    if (E.voxels) E.voxels.visible = section === "voxel";
    if (E.wire) E.wire.visible = section === "wire";
    if (section === "points" && !E.points && !mediaPts) rebuildCloud();
    if (section === "voxel" && !E.voxels) rebuildVoxels();
    if (section === "wire" && !E.wire) rebuildWire();
    if (section === "ascii" || section === "dither") { E.vw = 0; E.scene.fog = null; }
  }, [section, mediaPts, rebuildCloud, rebuildVoxels, rebuildWire]);

  useEffect(() => {
    if (section !== "voxel") return;
    rebuildVoxels();
  }, [section, modelName, vx.res, vx.gap, vx.colorSrc, vx.solid, vx.gradA, vx.gradB, vx.lit, rebuildVoxels]);

  useEffect(() => {
    const E = engine.current;
    if (!E || !E.voxels) return;
    placeVoxels(E.voxels, vx.scatter);
  }, [vx.scatter, cloudInfo.voxels]);

  useEffect(() => {
    if (section !== "wire") return;
    rebuildWire();
  }, [section, modelName, wf.mode, wf.angle, wf.hideBack, rebuildWire]);

  useEffect(() => {
    const E = engine.current;
    if (!E || !E.wire) return;
    const lm = E.wire.userData.lineMat, sm = E.wire.userData.solidMat;
    lm.color.set(wf.color);
    lm.opacity = wf.opacity;
    lm.blending = wf.additive ? THREE.AdditiveBlending : THREE.NormalBlending;
    lm.needsUpdate = true;
    sm.color.set(wf.bg);
    sm.needsUpdate = true;
  }, [wf.color, wf.opacity, wf.additive, wf.bg, cloudInfo.segs]);

  // resampling is the expensive path, so only these inputs trigger it
  useEffect(() => {
    if (section !== "points" || mediaPts) return;
    rebuildCloud();
  }, [section, mediaPts, modelName, pc.count, pc.colorSrc, pc.solid, pc.gradA, pc.gradB, rebuildCloud]);

  // these only touch existing buffers or material flags
  useEffect(() => {
    const E = engine.current;
    if (!E || !E.points) return;
    const g = E.points.geometry;
    const base = g.userData.base, dir = g.userData.dir;
    const arr = g.attributes.position.array;
    for (let i = 0; i < arr.length; i++) arr[i] = base[i] + dir[i] * pc.scatter;
    g.attributes.position.needsUpdate = true;
    g.computeBoundingSphere();
  }, [pc.scatter, cloudInfo.points]);

  useEffect(() => {
    const E = engine.current;
    if (!E || !E.points) return;
    const m = E.points.material;
    m.size = pc.size;
    m.sizeAttenuation = pc.atten;
    m.opacity = pc.opacity;
    m.map = pc.shape === "round" ? E.disc : null;
    m.alphaTest = pc.shape === "round" ? 0.4 : 0;
    m.blending = pc.additive ? THREE.AdditiveBlending : THREE.NormalBlending;
    m.depthWrite = !pc.additive;
    m.needsUpdate = true;
  }, [pc.size, pc.atten, pc.opacity, pc.shape, pc.additive, cloudInfo.points]);

  /* ---- model loading ---- */
  const swap = (built, name) => {
    const E = engine.current;
    if (E.content) {
      E.pivot.remove(E.content);
      E.content.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
      });
    }
    const bb = new THREE.Box3().setFromObject(built.group);
    E.half = bb.getSize(new THREE.Vector3()).multiplyScalar(0.5);
    E.content = built.group;
    E.pivot.add(built.group);
    triRef.current = built.tris;
    orbit.current.pitch = 0.18;
    orbit.current.yaw = (E.half.z > E.half.x ? Math.PI / 2 : 0) + 0.4;
    setModelName(name);
    applySurface(built.group, sRef.current);
    requestAnimationFrame(() => frameModel());
  };

  // Solve for the camera distance that keeps every corner of the bounding box
  // inside the frustum, sampled across a full turn so nothing clips while spinning.
  const frameModel = useCallback(() => {
    const E = engine.current, cv = canvasRef.current;
    if (!E || !E.half) return;
    const h = E.half.clone().multiplyScalar(sRef.current.scale);
    const aspect = cv && cv.clientHeight ? cv.clientWidth / cv.clientHeight : 1.6;
    const t = Math.tan((42 * Math.PI) / 360);
    const rot = new THREE.Matrix4(), v = new THREE.Vector3(), e = new THREE.Euler();
    let need = 0;
    for (let a = 0; a < 16; a++) {
      e.set(orbit.current.pitch, (a / 16) * Math.PI * 2, 0);
      rot.makeRotationFromEuler(e);
      for (let i = 0; i < 8; i++) {
        v.set(i & 1 ? h.x : -h.x, i & 2 ? h.y : -h.y, i & 4 ? h.z : -h.z).applyMatrix4(rot);
        need = Math.max(need, Math.abs(v.y) / t + v.z, Math.abs(v.x) / (t * aspect) + v.z);
      }
    }
    setS((p) => ({ ...p, zoom: Math.min(14, Math.max(1.4, need * 1.08)) }));
  }, []);

  const loadFile = async (file) => {
    if (!file) return;
    setError("");
    if (!/\.glb$/i.test(file.name)) {
      setError("Only .glb files work here. Export binary glTF from Blender, C4D, or Spline.");
      return;
    }
    if (file.size > 60 * 1024 * 1024) {
      setError("That file is over 60 MB. Decimate the mesh or export a lower LOD.");
      return;
    }
    setBusy(true);
    try {
      const buf = await file.arrayBuffer();
      const { json, bin } = parseGLB(buf);
      const textures = await loadTextures(json, bin);
      const built = buildModel(json, bin, textures);
      swap({ group: fitToUnitBox(built.group), tris: built.tris }, file.name.replace(/\.glb$/i, ""));
      const E = engine.current;
      if (E.points) { E.pivot.remove(E.points); E.points.geometry.dispose(); E.points.material.dispose(); E.points = null; }
      if (E.voxels) { E.pivot.remove(E.voxels); E.voxels.geometry.dispose(); E.voxels.material.dispose(); E.voxels = null; }
      if (E.wire) { E.pivot.remove(E.wire); disposeWire(E.wire); E.wire = null; }
      const sec = sectionRef.current;
      if (sec !== "ascii") E.content.visible = false;
      if (sec === "points") rebuildCloud();
      if (sec === "voxel") rebuildVoxels();
      if (sec === "wire") rebuildWire();
      flash("Model loaded");
    } catch (e) {
      setError(e.message || "That model couldn't be read.");
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    const E = engine.current;
    if (E && E.content) applySurface(E.content, s);
  }, [s.albedo, s.bump, s.spec, modelName]);

  /* ---- collage ---- */
  const drawCollage = useCallback(() => {
    const E = engine.current;
    const host = viewRef.current;
    // A still image only changes when something else does. Re-rendering it every
    // frame was burning the whole budget for an identical picture.
    const liveSrc = E.src && (E.src.kind === "camera" || E.src.kind === "video");
    if (!liveSrc && !E.capturing && host) {
      const key = JSON.stringify(clRef.current) + "|" + (E.src && E.src.kind) + "|" + (E.src && E.src.name) +
        "|" + host.clientWidth + "x" + host.clientHeight + "|" + E.cap + "|" + (E.alpha ? 1 : 0) +
        "|" + shotsRef.current.length + "|" + (E.shootCount || 0);
      if (key === E.clKey) return;
      E.clKey = key;
    } else E.clKey = null;
    const out = drawCollageFrame(E, canvasRef.current, clRef.current, viewRef.current, shotsRef.current);
    if (!out) return;
    if (E.shootCount > 0) {
      const cv = canvasRef.current;
      const ctx = cv.getContext("2d");
      const px = cv.width / (viewRef.current.clientWidth || 1);
      ctx.setTransform(px, 0, 0, px, 0, 0);
      const W = viewRef.current.clientWidth, H = viewRef.current.clientHeight;
      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "#fff";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.font = `700 ${Math.min(W, H) * 0.34}px ${MONO}`;
      ctx.fillText(String(E.shootCount), W / 2, H / 2);
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
    }
  }, []);

  const applyCollage = (k) => {
    setCl((p) => ({ ...p, ...CL_RECIPES[k] }));
    flash(k + " applied");
  };

  const clearShots = useCallback(() => {
    shotsRef.current.forEach((b) => b.close && b.close());
    shotsRef.current = [];
    setShoot({ busy: false, count: 0, have: 0 });
  }, []);

  // Photobooth: count down, then grab N stills spaced by interval.
  const runPhotobooth = useCallback(async () => {
    const E = engine.current;
    const el = E.src && E.src.el;
    if (!el || shoot.busy) return;
    const cfg = clRef.current;
    setShoot({ busy: true, count: 3, have: 0 });
    for (let c = 3; c > 0; c--) {
      E.shootCount = c;
      setShoot((p) => ({ ...p, count: c }));
      await new Promise((r) => setTimeout(r, 700));
    }
    E.shootCount = 0;
    clearShots();
    const grabbed = [];
    for (let i = 0; i < cfg.shots; i++) {
      try { grabbed.push(await createImageBitmap(el)); } catch {}
      shotsRef.current = grabbed.slice();
      setShoot({ busy: true, count: 0, have: grabbed.length });
      if (i < cfg.shots - 1) await new Promise((r) => setTimeout(r, cfg.interval));
    }
    setShoot({ busy: false, count: 0, have: grabbed.length });
    setCl((p) => ({ ...p, mode: "time" }));
    flash(grabbed.length + " frames captured");
  }, [shoot.busy, clearShots]);

  /* ---- media studio ---- */
  const drawMedia = useCallback(() => {
    const E = engine.current;
    const host = viewRef.current;
    const liveSrc = E.src && (E.src.kind === "camera" || E.src.kind === "video");
    // Same idea as the collage: idle on a still source instead of redrawing it.
    if (!liveSrc && !E.capturing && host && !mdRef.current.glitch && !mdRef.current.dust && !mdRef.current.grain) {
      const key = JSON.stringify(mdRef.current) + "|" + (E.src && E.src.kind) + "|" + (E.src && E.src.name) +
        "|" + host.clientWidth + "x" + host.clientHeight + "|" + E.cap + "|" + (E.alpha ? 1 : 0);
      if (key === E.mdKey) return;
      E.mdKey = key;
    } else E.mdKey = null;
    const out = drawMediaFrame(E, canvasRef.current, mdRef.current, viewRef.current);
    if (!out) {
      const cv = canvasRef.current;
      if (cv) {
        const ctx = cv.getContext("2d");
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.fillStyle = mdRef.current.bg;
        ctx.fillRect(0, 0, cv.width, cv.height);
      }
      return;
    }
    textRef.current = out.text;
    const L = lastStats.current;
    if (L.cols !== out.cols || L.rows !== out.rows) {
      lastStats.current = { ...L, cols: out.cols, rows: out.rows };
      if (!engine.current.capturing) setStats((p) => ({ ...p, cols: out.cols, rows: out.rows }));
    }
    const v = E.src && E.src.kind === "video" ? E.src.el : null;
    if (v && vidHeadRef.current && v.duration) {
      vidHeadRef.current.style.left = (v.currentTime / v.duration) * 100 + "%";
      if (vidClockRef.current)
        vidClockRef.current.textContent = v.currentTime.toFixed(1) + " / " + v.duration.toFixed(1) + "s";
    }
  }, []);

  const loadMediaFile = async (file) => {
    if (!file) return;
    setError("");
    if (/^video\//.test(file.type)) {
      const url = URL.createObjectURL(file);
      const video = document.createElement("video");
      video.src = url;
      video.loop = true;
      video.muted = true;
      video.playsInline = true;
      try {
        await new Promise((res, rej) => {
          video.onloadeddata = res;
          video.onerror = () => rej(new Error("That video couldn't be decoded. Try MP4 or WebM."));
        });
        await video.play();
        stopCamera();
        engine.current.src = { kind: "video", el: video, url };
        setSrc((p) => ({ ...p, kind: "video", name: file.name.replace(/\.[^.]+$/, "") }));
        setVid({ has: true, playing: true, dur: video.duration || 0, loop: true });
        flash(Math.round(video.videoWidth) + "×" + Math.round(video.videoHeight) + " video loaded");
      } catch (e) {
        URL.revokeObjectURL(url);
        setError(e.message || "That video couldn't be read.");
      }
      return;
    }
    setVid({ has: false, playing: false, dur: 0, loop: true });
    await loadImage(file);
  };

  const videoEl = () => {
    const E = engine.current;
    return E.src && E.src.kind === "video" ? E.src.el : null;
  };
  const toggleVideo = () => {
    const v = videoEl();
    if (!v) return;
    if (v.paused) { v.play(); setVid((p) => ({ ...p, playing: true })); }
    else { v.pause(); setVid((p) => ({ ...p, playing: false })); }
  };
  const seekVideo = (frac) => {
    const v = videoEl();
    if (!v || !v.duration) return;
    v.currentTime = Math.max(0, Math.min(v.duration, frac * v.duration));
  };

  const applyRecipe = (k) => {
    setMd((p) => ({ ...p, ...RECIPES[k] }));
    flash(k + " applied");
  };

  /* ---- hand control camera (separate from the source) ---- */
  const stopControlCam = useCallback(() => {
    const E = engine.current;
    if (!E || !E.ctrl) { setHandStatus("idle"); return; }
    try { E.ctrl.stream.getTracks().forEach((t) => t.stop()); } catch {}
    E.ctrl = null;
    E.hand = null;
  }, []);

  const startControlCam = useCallback(async () => {
    const E = engine.current;
    if (E.ctrl) return true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      const el = document.createElement("video");
      el.srcObject = stream;
      el.muted = true;
      el.playsInline = true;
      el.autoplay = true;
      await el.play();
      E.ctrl = { stream, el };
      setHandStatus("loading");
      try {
        await loadHandModel(E);
        setHandStatus("ready");
      } catch (e) {
        console.error("Hand model failed to load:", e);
        setHandStatus("error");
        setError("The hand tracking model could not be downloaded. Check your connection and try again.");
      }
      return true;
    } catch (e) {
      if (e && (e.name === "NotAllowedError" || e.name === "SecurityError")) {
        // The person said no. That is an answer, not an error.
        flash("Camera off \u2014 hand control needs it");
      } else {
        setError("Couldn't start the camera: " + ((e && e.message) || e));
      }
      setHc((p) => ({ ...p, on: false }));
      return false;
    }
  }, []);

  useEffect(() => {
    if (hc.on) startControlCam();
    else stopControlCam();
  }, [hc.on, startControlCam, stopControlCam]);

  useEffect(() => () => stopControlCam(), [stopControlCam]);



  /* ---- media sources ---- */
  const stopCamera = useCallback(() => {
    const E = engine.current;
    if (!E || !E.src) return;
    if (E.src.stream) E.src.stream.getTracks().forEach((t) => t.stop());
    if (E.src.kind === "video" && E.src.el) { try { E.src.el.pause(); } catch {} }
    if (E.src.url) URL.revokeObjectURL(E.src.url);
    E.src = null;
  }, []);

  const loadImage = async (file) => {
    if (!file) return;
    setError("");
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.decoding = "async";
      await new Promise((res, rej) => {
        img.onload = res;
        img.onerror = () => rej(new Error("That image couldn't be read."));
        img.src = url;
      });
      stopCamera();
      engine.current.src = { kind: "image", el: img, url };
      setSrc((p) => ({ ...p, kind: "image", name: file.name.replace(/\.[^.]+$/, "") }));
      flash(img.naturalWidth + "×" + img.naturalHeight + " loaded");
    } catch (e) {
      URL.revokeObjectURL(url);
      setError(e.message || "That image couldn't be read.");
    }
  };

  const startCamera = async () => {
    setError("");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError("This browser doesn't expose a camera API.");
      return;
    }
    setBusy(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      const video = document.createElement("video");
      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      video.autoplay = true;
      await video.play();
      stopCamera();
      engine.current.src = { kind: "camera", el: video, stream };
      setSrc((p) => ({ ...p, kind: "camera", name: "camera" }));
      flash("Camera live");
    } catch (e) {
      if (e && (e.name === "NotAllowedError" || e.name === "SecurityError"))
        flash("Camera off \u2014 allow it in the address bar to use this");
      else setError("Couldn't start the camera: " + ((e && e.message) || e));
      setSrc((p) => ({ ...p, kind: "model" }));
    } finally {
      setBusy(false);
    }
  };

  const pickSource = (kind) => {
    if (kind === src.kind) return;
    if (kind === "model") {
      stopCamera();
      setSrc((p) => ({ ...p, kind: "model" }));
    } else if (kind === "camera") {
      startCamera();
    } else {
      if (engine.current.src && engine.current.src.kind === "image")
        setSrc((p) => ({ ...p, kind: "image" }));
      else imgInputRef.current.click();
    }
  };

  useEffect(() => () => stopCamera(), [stopCamera]);

  // Leave the camera alone the moment the visible section stops needing it —
  // no point holding a live video track open behind a voxel grid.
  useEffect(() => {
    const E = engine.current;
    if (!E || !E.src) return;
    const uses = ["media", "collage", "ascii", "dither", "points"].includes(section);
    if (E.src.kind === "camera" && E.src.stream)
      E.src.stream.getVideoTracks().forEach((t) => (t.enabled = uses));
    if (E.src.kind === "video" && E.src.el) {
      if (!uses) E.src.el.pause();
      else if (vid.playing) E.src.el.play().catch(() => {});
    }
    E.clKey = null;
    E.mdKey = null;
  }, [section, vid.playing]);

  useEffect(() => {
    if (section !== "media" && section !== "collage") return;
    const onPaste = (e) => {
      const items = e.clipboardData && e.clipboardData.files;
      if (items && items.length) loadMediaFile(items[0]);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  });

  /* ---- pointer orbit ---- */
  useEffect(() => {
    const cv = viewRef.current;
    if (!cv) return;
    const down = (e) => {
      orbit.current.dragging = true;
      orbit.current.lx = e.clientX;
      orbit.current.ly = e.clientY;
      cv.setPointerCapture(e.pointerId);
    };
    const move = (e) => {
      if (!orbit.current.dragging) return;
      if (sectionRef.current === "media" || sectionRef.current === "collage") {
        // pan by mutating the ref directly; a state update per pointermove
        // would re-render the whole rail dozens of times a second
        const host = viewRef.current;
        const w = (host && host.clientWidth) || 1, h = (host && host.clientHeight) || 1;
        const dx = (e.clientX - orbit.current.lx) / w, dy = (e.clientY - orbit.current.ly) / h;
        const R = sectionRef.current === "media" ? mdRef : clRef;
        R.current = { ...R.current, panX: R.current.panX + dx, panY: R.current.panY + dy };
        orbit.current.lx = e.clientX;
        orbit.current.ly = e.clientY;
        return;
      }
      orbit.current.yaw += (e.clientX - orbit.current.lx) * 0.008;
      orbit.current.pitch = Math.max(-1.4, Math.min(1.4, orbit.current.pitch + (e.clientY - orbit.current.ly) * 0.008));
      orbit.current.lx = e.clientX;
      orbit.current.ly = e.clientY;
    };
    const up = () => {
      orbit.current.dragging = false;
      if (sectionRef.current === "media") setMd(mdRef.current); // commit the pan
      if (sectionRef.current === "collage") setCl(clRef.current);
    };
    const wheel = (e) => {
      e.preventDefault();
      const upd = (p) => ({ ...p, zoom: Math.min(14, Math.max(1.4, p.zoom + e.deltaY * 0.004)) });
      const sec = sectionRef.current;
      if (sec === "points") setPc(upd);
      else if (sec === "voxel") setVx(upd);
      else if (sec === "wire") setWf(upd);
      else if (sec === "dither") setDt(upd);
      else if (sec === "media") {
        setMd((p) => ({ ...p, zoom: Math.min(8, Math.max(0.15, p.zoom * (1 - e.deltaY * 0.0015))) }));
        return;
      }
      else if (sec === "collage") {
        setCl((p) => ({ ...p, zoom: Math.min(6, Math.max(0.15, p.zoom * (1 - e.deltaY * 0.0015))) }));
        return;
      }
      else setS(upd);
    };
    cv.addEventListener("pointerdown", down);
    cv.addEventListener("pointermove", move);
    cv.addEventListener("pointerup", up);
    cv.addEventListener("pointercancel", up);
    cv.addEventListener("wheel", wheel, { passive: false });
    return () => {
      cv.removeEventListener("pointerdown", down);
      cv.removeEventListener("pointermove", move);
      cv.removeEventListener("pointerup", up);
      cv.removeEventListener("pointercancel", up);
      cv.removeEventListener("wheel", wheel);
    };
  }, []);

  /* ---- exports ---- */
  const download = (blob, name) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    // Revoking while the browser is still writing a large file truncates it,
    // which is exactly how a "corrupt" download happens.
    setTimeout(() => URL.revokeObjectURL(url), 300000);
  };

  const exportPNG = () => {
    canvasRef.current.toBlob((b) => download(b, `${modelName}-ascii.png`), "image/png");
    flash("PNG saved");
  };

  const exportTXT = () => {
    download(new Blob([textRef.current], { type: "text/plain" }), `${modelName}-ascii.txt`);
    flash("Text saved");
  };

  const exportSVG = () => {
    if (s.mark === "dots") return exportDotSVG();
    const lines = textRef.current.split("\n");
    const cw = 8, ch = 8 * (s.lineHeight / 0.6);
    const w = (lines[0] || "").length * cw;
    const h = lines.length * ch;
    const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const body = lines
      .map((l, i) => `<text x="0" y="${(i + 0.8) * ch}" xml:space="preserve">${esc(l)}</text>`)
      .join("\n");
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><rect width="100%" height="100%" fill="${s.bg}"/><g font-family="monospace" font-size="${cw / 0.6}" fill="${s.tint}">${body}</g></svg>`;
    download(new Blob([svg], { type: "image/svg+xml" }), `${modelName}-ascii.svg`);
    flash("SVG saved");
  };

  const exportDotSVG = () => {
    const g = dotsRef.current;
    if (!g) return;
    const step = 10, R = step / 2;
    const w = g.cols * step, h = g.rows * step;
    const parts = [];
    for (let r = 0; r < g.rows; r++)
      for (let c = 0; c < g.cols; c++) {
        const k = r * g.cols + c;
        if (!g.hit[k]) continue;
        const rad = R * s.dot * Math.sqrt(g.lum[k]);
        if (rad < 0.2) continue;
        const fill = s.color === "mono" ? s.tint : g.colors[k];
        const cx = (c + 0.5) * step, cy = (r + 0.5) * step;
        parts.push(
          s.shape === "square"
            ? `<rect x="${(cx - rad).toFixed(2)}" y="${(cy - rad).toFixed(2)}" width="${(rad * 2).toFixed(2)}" height="${(rad * 2).toFixed(2)}" fill="${fill}"/>`
            : s.shape === "ring"
            ? `<circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="${rad.toFixed(2)}" fill="none" stroke="${fill}" stroke-width="${(rad * 0.56).toFixed(2)}"/>`
            : `<circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="${rad.toFixed(2)}" fill="${fill}"/>`
        );
      }
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><rect width="100%" height="100%" fill="${s.bg}"/>${parts.join("")}</svg>`;
    download(new Blob([svg], { type: "image/svg+xml" }), `${modelName}-dots.svg`);
    flash("SVG saved");
  };

  const exportPLY = () => {
    const E = engine.current;
    const P = E && (mediaPts ? E.mpts : E.points);
    if (!P) return;
    const pos = P.geometry.attributes.position.array;
    const col = P.geometry.attributes.color.array;
    const n = mediaPts ? P.geometry.drawRange.count : pos.length / 3;
    const rows = new Array(n);
    const srgb = (v) => Math.round(255 * (v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055));
    for (let i = 0; i < n; i++)
      rows[i] =
        pos[i * 3].toFixed(5) + " " + pos[i * 3 + 1].toFixed(5) + " " + pos[i * 3 + 2].toFixed(5) +
        " " + srgb(col[i * 3]) + " " + srgb(col[i * 3 + 1]) + " " + srgb(col[i * 3 + 2]);
    const header =
      "ply\nformat ascii 1.0\nelement vertex " + n +
      "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    download(new Blob([header + rows.join("\n") + "\n"], { type: "text/plain" }), `${modelName}-cloud.ply`);
    flash("PLY saved");
  };

  const exportScenePNG = () => {
    const E = engine.current;
    if (!E) return;
    if (sectionRef.current === "collage") {
      const prev = E.cap;
      E.cap = capFor();
      drawCollage();
      canvasRef.current.toBlob((b) => download(b, `${modelName}-collage.png`), "image/png");
      E.cap = prev;
      flash("PNG saved");
      return;
    }
    if (sectionRef.current === "media") {
      const prev = E.cap;
      E.cap = capFor();
      drawMedia();
      canvasRef.current.toBlob((b) => download(b, `${modelName}-media.png`), "image/png");
      E.cap = prev;
      flash("PNG saved");
      return;
    }
    draw3D(cfgFor(sectionRef.current), sectionRef.current);
    E.renderer.domElement.toBlob((b) => download(b, `${modelName}-${sectionRef.current}.png`), "image/png");
    flash("PNG saved");
  };


  const exportVoxelCSV = () => {
    const E = engine.current;
    if (!E || !E.voxels) return;
    const M = E.voxels;
    const m4 = new THREE.Matrix4(), v = new THREE.Vector3(), col = new THREE.Color();
    const step = M.geometry.parameters.width / (1 - vxRef.current.gap);
    const rows = ["x,y,z,hex"];
    for (let i = 0; i < M.count; i++) {
      M.getMatrixAt(i, m4);
      v.setFromMatrixPosition(m4);
      let hex = "ffffff";
      if (M.instanceColor) {
        col.fromArray(M.instanceColor.array, i * 3);
        hex = col.getHexString();
      }
      rows.push(
        Math.round(v.x / step) + "," + Math.round(v.y / step) + "," + Math.round(v.z / step) + ",#" + hex
      );
    }
    download(new Blob([rows.join("\n")], { type: "text/csv" }), `${modelName}-voxels.csv`);
    flash(M.count.toLocaleString() + " voxels saved");
  };

  const raf = () => new Promise((r) => requestAnimationFrame(r));

  // Export resolution is a target height, not a multiple of the window, so a
  // small browser window cannot quietly produce a small video.
  const capFor = () => {
    const host = viewRef.current;
    const h = host && host.clientHeight ? host.clientHeight : 720;
    return Math.max(1, Math.min(6, expRef.current.height / h));
  };

  const activeCanvas = () => {
    const E = engine.current;
    const sec = sectionRef.current;
    return sec === "ascii" || sec === "dither" || sec === "media" || sec === "collage"
      ? canvasRef.current
      : E.renderer.domElement;
  };

  const renderAt = (t, sec) => {
    applyTime(t);
    if (sec === "collage") drawCollage();
    else if (sec === "media") drawMedia();
    else if (sec === "ascii") draw(0);
    else if (sec === "dither") drawDither();
    else draw3D(cfgFor(sec), sec);
  };

  // Encode with WebCodecs and write the container by hand.
  //
  // MediaRecorder samples a live canvas against the wall clock, so a slow frame
  // becomes a duplicate and the catch-up becomes a jump — and its output has no
  // duration in the header, which is why players will not scrub it. Encoding
  // frame by frame with explicit timestamps fixes both.
  const recordLoop = async () => {
    if (recording) return;
    const E = engine.current;
    const D = tlRef.current.duration;
    const sec = sectionRef.current;
    const fps = expRef.current.fps;
    const frames = Math.round(D * fps);
    let encoder = null, rec = null;

    try {
      if (tlRef.current.seamless && !sectionKeys(sec).length) snapSpin(D);
      play.current.baseYaw = orbit.current.yaw;
      play.current.on = true;
      E.capturing = true;
      E.cap = capFor();
      E.bgOver = FIELDS[expRef.current.field];
      E.vw = 0;
      E.clKey = null;
      E.mdKey = null;
      setRecording(true);
      setPlaying(true);

      for (let w = 0; w < 6; w++) { renderAt((w / 6) * D, sec); await raf(); }
      renderAt(0, sec);
      await raf();

      const cv = activeCanvas();
      const W = cv.width, H = cv.height;
      const bits = Math.min(120000000, Math.round(W * H * Math.max(fps, 30) * 0.42));
      const prog = capRef.current;
      if (prog) prog.style.display = "flex";

      const canCodec =
        typeof VideoEncoder === "function" &&
        typeof VideoFrame === "function" &&
        VideoEncoder.isConfigSupported;

      let encoded = false;
      if (canCodec) {
       try {
        const config = {
          codec: avcCodec(W, H),
          width: W, height: H,
          bitrate: bits, framerate: fps,
          avc: { format: "avc" },
          latencyMode: "quality",
        };
        const support = await VideoEncoder.isConfigSupported(config);
        if (!support || !support.supported) throw new Error("H.264 encoding is not available here");

        const samples = [];
        let description = null, encErr = null;
        encoder = new VideoEncoder({
          output: (chunk, meta) => {
            if (!description && meta && meta.decoderConfig && meta.decoderConfig.description)
              description = new Uint8Array(meta.decoderConfig.description);
            const d = new Uint8Array(chunk.byteLength);
            chunk.copyTo(d);
            samples.push({ data: d, key: chunk.type === "key" });
          },
          error: (e) => (encErr = e),
        });
        encoder.configure(config);
        // Safari can accept a config and still not reach the configured state.
        if (encoder.state !== "configured") throw new Error("the encoder would not configure");

        const usec = 1000000 / fps;
        const gop = Math.max(1, Math.round(fps * 2));
        for (let i = 0; i < frames; i++) {
          renderAt((i / frames) * D, sec);
          const vf = new VideoFrame(cv, { timestamp: Math.round(i * usec), duration: Math.round(usec) });
          encoder.encode(vf, { keyFrame: i % gop === 0 });
          vf.close();
          if (prog) prog.textContent = "frame " + (i + 1) + " / " + frames;
          if (encErr) throw encErr;
          if (encoder.encodeQueueSize > 8) {
            while (encoder.encodeQueueSize > 4) await new Promise((r) => setTimeout(r, 4));
          } else if (i % 8 === 0) await raf();
        }
        await encoder.flush();
        encoder.close();
        encoder = null;
        if (encErr) throw encErr;

        if (prog) prog.textContent = "writing file\u2026";
        const mp4 = muxMP4({ samples, description, width: W, height: H, fps });
        const blob = new Blob([mp4], { type: "video/mp4" });
        download(blob, `${modelName}-${sec}-${D}s.mp4`);
        flash(`${(blob.size / 1048576).toFixed(1)} MB \u00b7 ${W}\u00d7${H} \u00b7 ${D}s \u00b7 ${frames} frames \u00b7 MP4`);
        encoded = true;
       } catch (codecErr) {
        console.warn("WebCodecs unavailable, falling back to the recorder:", codecErr);
        try { if (encoder && encoder.state !== "closed") encoder.close(); } catch {}
        encoder = null;
       }
      }

      if (!encoded) {
        let acc = 0, last = performance.now();
        for (let i = 0; i < 10; i++) { await raf(); const n = performance.now(); acc += n - last; last = n; }
        const refresh = Math.min(240, Math.max(50, Math.round(10000 / acc)));
        const step = Math.max(1, Math.round(refresh / fps));
        const real = Math.max(2, Math.round(D * (refresh / step)));
        const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
        const stream = cv.captureStream(0);
        const track = stream.getVideoTracks()[0];
        const canStep = track && typeof track.requestFrame === "function";
        const chunks = [];
        rec = new MediaRecorder(canStep ? stream : cv.captureStream(fps),
          { mimeType: mime, videoBitsPerSecond: bits });
        rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
        const finished = new Promise((res) => (rec.onstop = res));
        rec.start(1000);
        for (let i = 0; i < real; i++) {
          for (let k = 0; k < step; k++) await raf();
          renderAt((i / real) * D, sec);
          if (canStep) track.requestFrame();
          if (prog) prog.textContent = "frame " + (i + 1) + " / " + real;
        }
        await new Promise((r) => setTimeout(r, 1000 / fps + 180));
        if (rec.state !== "inactive") rec.requestData();
        await new Promise((r) => setTimeout(r, 120));
        rec.stop();
        await finished;
        const ext = (rec.mimeType || mime).indexOf("mp4") >= 0 ? "mp4" : "webm";
        const blob = new Blob(chunks, { type: rec.mimeType || mime });
        if (blob.size < 2048) throw new Error("the encoder produced only " + blob.size + " bytes");
        download(blob, `${modelName}-${sec}-${D}s.${ext}`);
        flash(`${(blob.size / 1048576).toFixed(1)} MB \u00b7 ${ext.toUpperCase()} \u00b7 recorder fallback`);
      }
    } catch (err) {
      console.error("Video capture failed:", err);
      try { if (encoder && encoder.state !== "closed") encoder.close(); } catch {}
      try { if (rec && rec.state !== "inactive") rec.stop(); } catch {}
      setError("Video capture failed: " + ((err && err.message) || err) +
        "\n\nThe full error is in DevTools \u2192 Console.");
    } finally {
      E.capturing = false;
      E.cap = 0;
      E.bgOver = null;
      E.vw = 0;
      E.clKey = null;
      E.mdKey = null;
      play.current.on = false;
      setRecording(false);
      setPlaying(false);
      if (capRef.current) { capRef.current.style.display = "none"; capRef.current.textContent = ""; }
      setSeq({ busy: false, done: 0, total: 0 });
      syncBack();
    }
  };

  const exportSequence = async (_fps, alpha) => {
    const fps = expRef.current.fps;
    const E = engine.current;
    const sec = sectionRef.current;
    const D = tlRef.current.duration;
    const total = Math.round(D * fps);
    if (tlRef.current.seamless && !sectionKeys(sec).length) snapSpin(D);
    play.current.baseYaw = orbit.current.yaw;
    play.current.on = false;
    setPlaying(false);
    E.alpha = !!alpha;
    E.cap = capFor();
    E.capturing = true;
    E.vw = 0;
    setSeq({ busy: true, done: 0, total });
    const files = [];
    try {
      for (let i = 0; i < total; i++) {
        renderAt((i / total) * D, sec);
        const cv = activeCanvas();
        const blob = await new Promise((r) => cv.toBlob(r, "image/png"));
        files.push({
          name: `${sec}_${String(i).padStart(4, "0")}.png`,
          data: new Uint8Array(await blob.arrayBuffer()),
        });
        if (i % 3 === 0) {
          setSeq({ busy: true, done: i + 1, total });
          await new Promise((r) => setTimeout(r, 0));
        }
      }
      download(zipStore(files), `${modelName}-${sec}-${fps}fps.zip`);
      flash(total + " frames saved");
    } catch (e) {
      setError("Sequence export failed: " + (e.message || e));
    } finally {
      E.alpha = false;
      E.cap = 0;
      E.capturing = false;
      E.vw = 0;
      if (capRef.current) { capRef.current.style.display = "none"; capRef.current.textContent = ""; }
      setSeq({ busy: false, done: 0, total: 0 });
      syncBack();
    }
  };


  const exportDitherPNG = (raw) => {
    const E = engine.current;
    if (!E) return;
    if (raw && E.dcv) {
      E.dcv.toBlob((b) => download(b, `${modelName}-dither-1x.png`), "image/png");
      flash(`${E.dcv.width}×${E.dcv.height} saved`);
      return;
    }
    canvasRef.current.toBlob((b) => download(b, `${modelName}-dither.png`), "image/png");
    flash("PNG saved");
  };
  const exportWireSVG = async () => {
    const E = engine.current;
    if (!E || !E.wire) { setError("Nothing to plot yet."); return; }
    setBusy(true);
    await new Promise((r) => setTimeout(r, 30)); // let the button state paint
    try {
      const t0 = performance.now();
      const paths = plotterPaths(E, plot);
      if (!paths) throw new Error("no geometry");
      const out = plotterSVG(paths, plot, E.camera.aspect || 1.5);
      if (!out) throw new Error("nothing visible from this angle");
      download(new Blob([out.svg], { type: "image/svg+xml" }), `${modelName}-plot-${plot.paper}.svg`);
      setPlotInfo({
        paths: out.paths,
        ink: out.ink,
        minutes: out.minutes,
        edges: paths.edgeCount,
        ms: Math.round(performance.now() - t0),
      });
      flash(
        `${out.paths.toLocaleString()} pen strokes \u00b7 ${(out.ink / 1000).toFixed(1)} m of ink \u00b7 about ${
          out.minutes < 1 ? "under a minute" : Math.round(out.minutes) + " min"
        } to plot`
      );
    } catch (err) {
      console.error("Plot export failed:", err);
      setError("Plot export failed: " + ((err && err.message) || err));
    } finally {
      setBusy(false);
    }
  };

  const copyText = async () => {
    try {
      await navigator.clipboard.writeText(textRef.current);
      flash("Copied to clipboard");
    } catch {
      const ta = document.createElement("textarea");
      ta.value = textRef.current;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      flash("Copied to clipboard");
    }
  };

  const copySettings = async () => {
    const json = JSON.stringify(s, null, 2);
    try {
      await navigator.clipboard.writeText(json);
      flash("Settings copied as JSON");
    } catch {
      flash("Clipboard blocked here");
    }
  };


  const applyTheme = (k) => setS((p) => ({ ...p, ...THEMES[k] }));

  const ramp = (s.ramp === "custom" ? s.custom : RAMPS[s.ramp]) || RAMPS.standard;

  const HandPanel = () => (
    <div className="grp">
      <h3>Hand control</h3>
      <Toggle label="Steer with your hand" on={hc.on} set={(v) => setH("on", v)} />
      {hc.on && (
        <>
          <div style={{ display: "flex", gap: 10, alignItems: "center", margin: "10px 0 8px" }}>
            <canvas
              ref={handCvRef}
              style={{
                width: 120, height: 90, borderRadius: 8, border: "1px solid var(--line2)",
                flex: "none", background: "#101418",
              }}
            />
            <div style={{ fontSize: 12, lineHeight: 1.5, color: handSeen ? "var(--accent)" : "var(--faint)" }}>
              {handStatus === "loading"
                ? "Downloading the hand model\u2026"
                : handStatus === "error"
                ? "Model failed to load."
                : handSeen
                ? "Hand tracked."
                : "Hold a hand up to the camera."}
              <div style={{ color: "var(--faint)", fontSize: 11, marginTop: 3 }}>
                {handStatus === "loading"
                  ? "About 3 MB, once per visit."
                  : "It follows the hand only \u2014 faces and background are ignored."}
              </div>
            </div>
          </div>

          <p className="hint" style={{ margin: "4px 0 12px" }}>
            <b style={{ color: "var(--text)" }}>Move left and right</b> to turn it.<br />
            <b style={{ color: "var(--text)" }}>Open your hand</b> to scatter, <b style={{ color: "var(--text)" }}>make a fist</b> to pull it back.<br />
            <b style={{ color: "var(--text)" }}>Pinch thumb and finger</b> to shrink it, spread them to grow it.
          </p>

          <Toggle label="Turn" on={hc.rotate} set={(v) => setH("rotate", v)} />
          <Toggle label="Scatter" on={hc.scatter} set={(v) => setH("scatter", v)} />
          <Toggle label="Scale" on={hc.scale} set={(v) => setH("scale", v)} />
          <Slide label="Sensitivity" v={hc.sensitivity} min={0.4} max={2} step={0.01} on={(v) => setH("sensitivity", v)} />
          <Slide label="Smoothing" v={hc.smooth} min={0} max={0.94} step={0.01} on={(v) => setH("smooth", v)} />
          <p className="hint">
            The model only downloads when you switch this on, so it costs nothing on a normal
            visit. Your model stays on screen \u2014 the camera is only the controller.
          </p>
        </>
      )}
    </div>
  );

  /* ---- UI ---- */
  return (
    <div className="gw">
      <style>{`
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

.gw{
  --void:#0B0D10; --panel:#101419; --sunk:#0D1116; --line:#1F262F; --line2:#2C3742;
  --text:#D6DDE5; --dim:#78889A; --faint:#4A5765;
  --accent:#5B8CFF; --sulfur:#E8B44A; --bad:#FF6B5E;
  position:absolute; inset:0; display:flex; flex-direction:column;
  background:var(--void); color:var(--text);
  font-family:"IBM Plex Sans",system-ui,sans-serif; font-size:13px;
  overflow:hidden; -webkit-font-smoothing:antialiased;
}
.gw *{box-sizing:border-box;}
.mono{font-family:${MONO};}

/* top bar */
.bar{display:flex;align-items:center;gap:14px;padding:0 14px;height:46px;
  border-bottom:1px solid var(--line);background:var(--panel);flex:none;}
.mark{font-weight:700;letter-spacing:-0.02em;font-size:15px;}
.mark span{color:var(--accent);}
.tag{font-family:${MONO};font-size:10px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--faint);border:1px solid var(--line2);padding:3px 6px;border-radius:2px;}
.spacer{flex:1;}

.btn{font-family:${MONO};font-size:11px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--text);background:var(--sunk);border:1px solid var(--line2);
  padding:6px 10px;border-radius:2px;cursor:pointer;transition:.14s;white-space:nowrap;}
.btn:hover{border-color:var(--accent);color:#fff;}
.btn:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
.btn.pri{background:var(--accent);border-color:var(--accent);color:#06080C;font-weight:600;}
.btn.pri:hover{background:#7BA3FF;}
.btn[disabled]{opacity:.45;cursor:not-allowed;}

/* body */
.body{flex:1;display:flex;min-height:0;}
.stage{flex:1;position:relative;min-width:0;display:flex;flex-direction:column;}
.view{flex:1;position:relative;min-height:0;cursor:grab;touch-action:none;}
.view:active{cursor:grabbing;}
.view>canvas,.view>.gl{position:absolute;inset:0;width:100%;height:100%;}
canvas.out{display:block;}
.gl canvas{display:block;width:100%!important;height:100%!important;}

.tabs{display:flex;border:1px solid var(--line2);border-radius:2px;
  flex:0 1 auto;min-width:0;overflow-x:auto;
  scrollbar-width:none;-webkit-overflow-scrolling:touch;}
.tabs::-webkit-scrollbar{display:none;}
.tabs button{flex:0 0 auto;font-family:${MONO};font-size:10px;letter-spacing:.08em;text-transform:uppercase;
  padding:6px 9px;background:var(--sunk);color:var(--dim);border:0;border-right:1px solid var(--line2);cursor:pointer;}
.tabs button:last-child{border-right:0;}
.tabs button.on{background:var(--accent);color:#06080C;font-weight:600;}
.tabs button:hover:not(.on){color:var(--text);}

.drop{position:absolute;inset:10px;border:1px dashed var(--accent);border-radius:3px;
  background:rgba(91,140,255,.08);display:flex;align-items:center;justify-content:center;
  font-family:${MONO};font-size:12px;letter-spacing:.1em;text-transform:uppercase;pointer-events:none;}

.timeline{flex:none;display:flex;align-items:center;gap:8px;padding:0 10px;height:40px;
  border-top:1px solid var(--line);background:var(--panel);
  overflow-x:auto;scrollbar-width:none;-webkit-overflow-scrolling:touch;}
.timeline::-webkit-scrollbar{display:none;}
.timeline > *{flex:0 0 auto;}
.timeline .track{flex:1 1 90px;}
.tbtn{font-family:${MONO};font-size:10px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--text);background:var(--sunk);border:1px solid var(--line2);
  padding:5px 9px;border-radius:2px;cursor:pointer;white-space:nowrap;}
.tbtn:hover{border-color:var(--accent);}
.tbtn.pri{background:var(--accent);border-color:var(--accent);color:#06080C;font-weight:600;}
.track{flex:1;position:relative;height:22px;background:var(--sunk);border:1px solid var(--line2);
  border-radius:2px;cursor:pointer;min-width:80px;touch-action:none;}
.track .kf{position:absolute;top:4px;width:8px;height:8px;margin-left:-4px;
  background:var(--sulfur);transform:rotate(45deg);pointer-events:none;}
.track .head{position:absolute;top:-1px;bottom:-1px;width:2px;margin-left:-1px;
  background:var(--accent);pointer-events:none;left:0;}
.clock{font-size:11px;color:var(--dim);min-width:46px;text-align:right;}
.tseg{margin-bottom:0;}
.tseg button{padding:5px 7px;}
@media (max-width:820px){
  .timeline{gap:5px;padding:0 6px;}
  .clock{display:none;}
  .bar{padding:0 8px;gap:8px;}
  .rail{max-height:44%;}
  .stage{min-height:52%;}
}

/* A phone on its side is wide but very short. Stacking the rail underneath
   leaves almost no height for the picture, so put it back alongside. */
@media (orientation:landscape) and (max-height:600px){
  .body{flex-direction:row;}
  .rail{width:min(46%,320px);max-height:none;border-top:0;border-left:1px solid var(--line);}
  .stage{min-height:0;}
  .bar{height:42px;}
  .timeline{height:38px;}
  .status{height:26px;}
}

.status{flex:none;height:28px;display:flex;align-items:center;gap:0;
  border-top:1px solid var(--line);background:var(--panel);font-family:${MONO};font-size:10.5px;color:var(--dim);}
.status i{font-style:normal;padding:0 12px;border-right:1px solid var(--line);height:100%;display:flex;align-items:center;}
.status i.hi{color:var(--text);}
.status .push{flex:1;border:0;}

.help{position:absolute;left:14px;bottom:92px;z-index:20;width:min(420px,calc(100% - 28px));
  background:var(--panel);border:1px solid var(--line2);border-radius:3px;padding:13px 15px 15px;
  box-shadow:0 18px 44px rgba(0,0,0,.55);font-size:12.5px;line-height:1.6;color:var(--dim);}
.help .hhead{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px;}
.help .hhead b{color:var(--text);font-size:13px;letter-spacing:-0.01em;}
.help ol{margin:0 0 10px;padding-left:17px;}
.help li{margin-bottom:5px;}
.help b{color:var(--text);font-weight:600;}
.help .ex{margin:0 0 8px;padding:8px 10px;background:var(--sunk);border-left:2px solid var(--sulfur);
  border-radius:0 2px 2px 0;font-size:12px;}
.help .ex:last-child{margin-bottom:0;border-left-color:var(--line2);}

.err{position:absolute;left:14px;right:14px;bottom:92px;z-index:20;max-height:46%;overflow:auto;background:#1A1012;
  border:1px solid var(--bad);border-left-width:3px;color:#FFC9C3;padding:9px 12px;
  font-size:12px;border-radius:2px;display:flex;gap:10px;align-items:flex-start;}
.err b{color:var(--bad);font-family:${MONO};font-size:10px;letter-spacing:.1em;text-transform:uppercase;}
.toast{position:absolute;top:14px;left:50%;transform:translateX(-50%);
  background:var(--panel);border:1px solid var(--line2);padding:6px 12px;border-radius:2px;
  font-family:${MONO};font-size:11px;letter-spacing:.06em;}

/* rail */
.rail{width:286px;flex:none;border-left:1px solid var(--line);background:var(--panel);
  overflow-y:auto;overscroll-behavior:contain;}
.rail::-webkit-scrollbar{width:8px;}
.rail::-webkit-scrollbar-thumb{background:var(--line2);}
.grp{border-bottom:1px solid var(--line);padding:12px 14px 15px;}
.grp h3{margin:0 0 11px;font-family:${MONO};font-size:10px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--faint);display:flex;align-items:center;gap:8px;}
.grp h3:after{content:"";flex:1;height:1px;background:var(--line);}

.row{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px;gap:8px;}
.row label{font-size:12px;color:var(--dim);}
.val{font-family:${MONO};font-size:11px;color:var(--text);min-width:44px;text-align:right;}

input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:16px;background:transparent;cursor:pointer;margin:0 0 12px;}
input[type=range]::-webkit-slider-runnable-track{height:2px;background:var(--line2);}
input[type=range]::-moz-range-track{height:2px;background:var(--line2);}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:3px;height:14px;
  background:var(--accent);border:0;border-radius:0;margin-top:-6px;}
input[type=range]::-moz-range-thumb{width:3px;height:14px;background:var(--accent);border:0;border-radius:0;}
input[type=range]:focus-visible{outline:1px solid var(--accent);outline-offset:4px;}

.seg{display:flex;border:1px solid var(--line2);border-radius:2px;overflow:hidden;margin-bottom:11px;}
.seg button{flex:1;font-family:${MONO};font-size:10px;letter-spacing:.05em;text-transform:uppercase;
  padding:6px 2px;background:var(--sunk);color:var(--dim);border:0;border-right:1px solid var(--line2);cursor:pointer;}
.seg button:last-child{border-right:0;}
.seg button.on{background:var(--accent);color:#06080C;font-weight:600;}
.seg button:hover:not(.on){color:var(--text);}

.chips{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px;}
.chip{font-family:${MONO};font-size:10px;padding:4px 7px;background:var(--sunk);
  border:1px solid var(--line2);color:var(--dim);cursor:pointer;border-radius:2px;}
.chip.on{border-color:var(--accent);color:var(--accent);}

.tog{display:flex;align-items:center;justify-content:space-between;padding:5px 0;cursor:pointer;}
.tog span{font-size:12px;color:var(--dim);}
.sw{width:30px;height:16px;background:var(--sunk);border:1px solid var(--line2);border-radius:2px;position:relative;flex:none;}
.sw i{position:absolute;top:2px;left:2px;width:10px;height:10px;background:var(--faint);transition:.14s;}
.tog.on .sw{border-color:var(--accent);}
.tog.on .sw i{left:16px;background:var(--accent);}

.swatches{display:flex;gap:6px;margin-bottom:10px;}
.sw2{width:100%;height:26px;border:1px solid var(--line2);border-radius:2px;cursor:pointer;position:relative;overflow:hidden;}
.sw2 input{position:absolute;inset:-4px;width:200%;height:200%;border:0;padding:0;cursor:pointer;background:none;}

.ramp{border:1px solid var(--line2);border-radius:2px;overflow:hidden;margin-bottom:10px;}
.ramp .strip{font-family:${MONO};font-size:13px;letter-spacing:1px;padding:9px 8px;
  white-space:nowrap;overflow:hidden;text-align:center;}
.ramp input{width:100%;font-family:${MONO};font-size:11px;background:var(--sunk);
  border:0;border-top:1px solid var(--line2);color:var(--text);padding:6px 8px;outline:none;}

.exports{display:grid;grid-template-columns:1fr 1fr;gap:6px;}
.exports .wide{grid-column:1/-1;}
.hint{font-size:11px;color:var(--faint);line-height:1.5;margin:9px 0 0;}

@media (max-width:820px){
  .body{flex-direction:column;}
  .rail{width:100%;border-left:0;border-top:1px solid var(--line);max-height:46%;}
  .stage{min-height:44%;}
  .hideSm{display:none;}
}
@media (prefers-reduced-motion:reduce){.sw i{transition:none;}}
      `}</style>

      <div className="bar">
        <div className="mark">GLYPH<span>WORKS</span></div>
        <div className="tabs">
          {[["media", "Media"], ["collage", "Collage"], ["ascii", "ASCII"], ["dither", "Dither"], ["points", "Points"], ["voxel", "Voxel"], ["wire", "Wire"]].map(([k, l]) => (
            <button key={k} className={section === k ? "on" : ""} onClick={() => setSection(k)}>{l}</button>
          ))}
        </div>
        <div className="tag hideSm">{BUILD}</div>
        <div className="spacer" />
        {(section === "ascii" || section === "dither" || section === "points" || section === "collage") && (
          <div className="tabs hideSm">
            {[["model", "Model"], ["image", "Image"], ["camera", "Camera"]].map(([k, l]) => (
              <button key={k} className={src.kind === k ? "on" : ""} onClick={() => pickSource(k)}>{l}</button>
            ))}
          </div>
        )}
        <button
          className="btn pri"
          onClick={() => (section === "media" || section === "collage"
            ? mediaInputRef.current.click()
            : src.kind === "image" && mediaOn
            ? imgInputRef.current.click()
            : fileRef.current.click())}
          disabled={busy}
        >
          {busy ? "Reading…" : section === "media" || section === "collage" ? "Open photo or video" : src.kind === "image" && mediaOn ? "Open image" : "Upload .glb"}
        </button>
        <button className="btn hideSm" onClick={() => setPanelOpen((v) => !v)}>
          {panelOpen ? "Hide controls" : "Controls"}
        </button>
        <input
          ref={mediaInputRef}
          type="file"
          accept="image/*,video/*"
          style={{ display: "none" }}
          onChange={(e) => { loadMediaFile(e.target.files[0]); e.target.value = ""; }}
        />
        <input
          ref={imgInputRef}
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => { loadImage(e.target.files[0]); e.target.value = ""; }}
        />
        <input
          ref={fileRef}
          type="file"
          accept=".glb"
          style={{ display: "none" }}
          onChange={(e) => loadFile(e.target.files[0])}
        />
      </div>

      <div className="body">
        <div
          className="stage"
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const f = e.dataTransfer.files[0];
            if (section === "media" || section === "collage") loadMediaFile(f);
            else if (f && /^image\//.test(f.type) && (section === "ascii" || section === "dither" || section === "points")) loadImage(f);
            else loadFile(f);
          }}
        >
          <div className="view" ref={viewRef}>
            <canvas ref={canvasRef} className="out" style={{ display: section === "ascii" || section === "dither" || section === "media" || section === "collage" ? "block" : "none" }} />
            <div className="gl" ref={glRef} style={{ display: section === "ascii" || section === "dither" || section === "media" || section === "collage" ? "none" : "block" }} />
          </div>
          {dragOver && <div className="drop">{section === "media" || section === "collage" ? "Drop a photo or a video" : section === "wire" || section === "voxel" ? "Drop the .glb" : "Drop a .glb or an image"}</div>}
          {toast && <div className="toast">{toast}</div>}
          {helpOpen && (
            <div className="help">
              <div className="hhead">
                <b>Animating in four steps</b>
                <button className="tbtn" onClick={() => setHelpOpen(false)}>Close</button>
              </div>
              <ol>
                <li>Park the playhead. Press <b>Stop</b> for 0s, or drag the track.</li>
                <li>Set the look you want at that moment — sliders, colours, drag the model to orbit.</li>
                <li>Press <b>+ Key</b>. A yellow diamond drops on the track.</li>
                <li>Move the playhead elsewhere, change the look, press <b>+ Key</b> again. Then <b>Play</b>.</li>
              </ol>
              <p className="ex">
                <b>Worked example — a voxel bloom.</b> Open <b>Voxel</b>. Stop, set Scatter to 0, + Key.
                Drag the playhead to 5s, set Scatter to 0.35, + Key. Drag to 10s, set Scatter back to 0,
                + Key. Play. The blocks burst apart and reassemble on a loop.
              </p>
              <p className="ex">
                <b>Getting video into After Effects without PNGs.</b> Set the export field to
                <b> Black</b>, record the MP4, drop it in AE and set the layer blend mode to
                <b> Screen</b> or <b>Add</b>. Pure black vanishes and anti-aliased edges survive
                intact — no keying, one file. Use <b>Green</b> instead for solid voxels, where
                blend modes won't work.
              </p>
              <p className="ex">
                Each key stores the <b>whole look</b> at that instant, not one slider. Two keys is the
                minimum for motion — one key alone just holds a pose. Count, Resolution and Crease
                angle can't animate, because changing them rebuilds the geometry.
              </p>
            </div>
          )}
          {error && (
            <div className="err">
              <b>Error</b>
              <div style={{ flex: 1 }}>{error}</div>
              <button className="btn" onClick={() => setError("")}>Dismiss</button>
            </div>
          )}
          {section === "media" && vid.has && (
            <div className="timeline">
              <button className="tbtn pri" onClick={toggleVideo}>{vid.playing ? "Pause" : "Play"}</button>
              <button className="tbtn" onClick={() => { seekVideo(0); }}>Restart</button>
              <div className="track"
                onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId);
                  const r = e.currentTarget.getBoundingClientRect(); seekVideo((e.clientX - r.left) / r.width); }}
                onPointerMove={(e) => { if (e.buttons !== 1) return;
                  const r = e.currentTarget.getBoundingClientRect(); seekVideo((e.clientX - r.left) / r.width); }}>
                <span className="head" ref={vidHeadRef} />
              </div>
              <span className="clock mono" ref={vidClockRef}>0.0s</span>
              <button className="tbtn hideSm" onClick={() => { const v = videoEl(); if (v) { v.loop = !v.loop; setVid((p) => ({ ...p, loop: v.loop })); } }}>
                {vid.loop ? "Loop on" : "Loop off"}
              </button>
            </div>
          )}
          <div className="timeline">
            <button className="tbtn pri" onClick={() => {
              if (play.current.on) { play.current.on = false; setPlaying(false); syncBack(); }
              else { play.current.baseYaw = orbit.current.yaw; play.current.on = true; setPlaying(true); }
            }}>{playing ? "Pause" : "Play"}</button>
            <button className="tbtn" onClick={() => { play.current.on = false; setPlaying(false); seekTo(0); syncBack(); }}>Stop</button>

            <div className="track" ref={trackRef}
              onPointerDown={(e) => {
                e.currentTarget.setPointerCapture(e.pointerId);
                const r = e.currentTarget.getBoundingClientRect();
                seekTo(((e.clientX - r.left) / r.width) * tl.duration);
              }}
              onPointerMove={(e) => {
                if (e.buttons !== 1) return;
                const r = e.currentTarget.getBoundingClientRect();
                seekTo(((e.clientX - r.left) / r.width) * tl.duration);
              }}
              onPointerUp={() => { if (!play.current.on) syncBack(); }}>
              {tl.keys.filter((k) => k.section === section).map((k) => (
                <span key={k.id} className="kf" style={{ left: (k.t / tl.duration) * 100 + "%" }} />
              ))}
              <span className="head" ref={headRef} />
            </div>

            <span className="clock mono" ref={clockRef}>0.00s</span>
            <button className="tbtn" onClick={addKey} disabled={liveSource}>+ Key</button>
            <button className={"tbtn" + (helpOpen ? " pri" : "")} onClick={() => setHelpOpen((v) => !v)}>?</button>
            <button className="tbtn" onClick={() => setTl((p) => ({
              ...p, keys: p.keys.filter((k) => !(k.section === section && Math.abs(k.t - play.current.t) < 0.15)) }))}>Del</button>
            <button className="tbtn hideSm" onClick={() => setTl((p) => ({ ...p, keys: p.keys.filter((k) => k.section !== section) }))}>Clear</button>
            <div className="seg tseg">
              {[5, 10, 15].map((d) => (
                <button key={d} className={tl.duration === d ? "on" : ""} onClick={() => setTl((p) => ({ ...p, duration: d }))}>{d}s</button>
              ))}
            </div>
            <div className="seg tseg hideSm" title="Export resolution">
              {[720, 1080, 1440, 2160].map((q) => (
                <button key={q} className={exp.height === q ? "on" : ""} onClick={() => setExp((p) => ({ ...p, height: q }))}>{q === 2160 ? "4K" : q + "p"}</button>
              ))}
            </div>
            <div className="seg tseg hideSm" title="Export background">
              {[["scene", "Field"], ["black", "Black"], ["white", "White"], ["green", "Green"]].map(([k, l]) => (
                <button key={k} className={exp.field === k ? "on" : ""} onClick={() => setExp((p) => ({ ...p, field: k }))}>{l}</button>
              ))}
            </div>
            <div className="seg tseg hideSm" title="Export frame rate">
              {[30, 60].map((f) => (
                <button key={f} className={exp.fps === f ? "on" : ""} onClick={() => setExp((p) => ({ ...p, fps: f }))}>{f}</button>
              ))}
            </div>
            <div className="seg tseg hideSm">
              {["linear", "smooth", "in", "out"].map((e) => (
                <button key={e} className={tl.ease === e ? "on" : ""} onClick={() => setTl((p) => ({ ...p, ease: e }))}>{e}</button>
              ))}
            </div>
          </div>

          <div className="status">
            <i className="hi">{section === "media" || mediaOn ? src.name || src.kind : modelName}</i>
            {section === "ascii" || section === "dither" ? (
              <>
                <i>{stats.cols} × {stats.rows}</i>
                <i>{(stats.cols * stats.rows).toLocaleString()} px</i>
              </>
            ) : section === "collage" ? (
              <i>{cl.layout === "mirror" ? "4 reflections"
                 : cl.layout === "kaleido" ? cl.wedges + " wedges"
                 : (cl.layout === "strip" ? cl.rows : cl.layout === "split" ? cl.cols : cl.cols * cl.rows) + " panels"}</i>
            ) : section === "points" ? (
              <i>{cloudInfo.points.toLocaleString()} points</i>
            ) : section === "voxel" ? (
              <i>{cloudInfo.voxels.toLocaleString()} voxels</i>
            ) : (
              <i>{cloudInfo.segs.toLocaleString()} edges</i>
            )}
            <i className="hideSm">{stats.tris.toLocaleString()} tris</i>
            <i className="hideSm">
              {(() => {
                const n = tl.keys.filter((k) => k.section === section).length;
                return n === 1 ? "1 key — add another for motion" : n + " keys";
              })()}
            </i>
            <span className="push" />
            <i className="hi" ref={capRef} style={{ display: "none" }} />
            {seq.busy && <i className="hi">frame {seq.done} / {seq.total}</i>}
            <i>{stats.fps} fps</i>
          </div>
        </div>

        {panelOpen && section === "points" && (
          <div className="rail">
            <HandPanel />
            <div className="grp">
              <h3>Source</h3>
              <div className="seg">
                {[["model", "Model"], ["image", "Image"], ["camera", "Camera"]].map(([k, l]) => (
                  <button key={k} className={src.kind === k ? "on" : ""} onClick={() => pickSource(k)}>{l}</button>
                ))}
              </div>
              {mediaPts && (
                <>
                  <Slide label="Grid" v={pc.mres} min={40} max={320} step={2}
                    fmt={(v) => v + " wide"} on={(v) => setP("mres", v)} />
                  <Slide label="Relief depth" v={pc.mdepth} min={0} max={2} step={0.01} on={(v) => setP("mdepth", v)} />
                  <Slide label="Cutout" v={src.cut} min={0} max={0.9} step={0.005}
                    fmt={(v) => (v ? v.toFixed(3) : "off")} on={(v) => setSrc((p) => ({ ...p, cut: v }))} />
                  <Toggle label="Invert depth" on={pc.minv} set={(v) => setP("minv", v)} />
                  <Toggle label="Mirror" on={src.mirror} set={(v) => setSrc((p) => ({ ...p, mirror: v }))} />
                  <p className="hint">
                    One point per sampled pixel, pushed along Z by brightness. Drag to orbit and the
                    picture turns into a landscape. Bright areas come toward you — invert to sink them.
                  </p>
                </>
              )}
            </div>

            <div className="grp">
              <h3>Points</h3>
              {!mediaPts && (
                <Slide label="Count" v={pc.count} min={2000} max={200000} step={1000}
                  fmt={(v) => (v / 1000).toFixed(0) + "k"} on={(v) => setP("count", v)} />
              )}
              <Slide label="Size" v={pc.size} min={0.004} max={0.16} step={0.001}
                fmt={(v) => v.toFixed(3)} on={(v) => setP("size", v)} />
              <Slide label="Anti-flicker" v={pc.minPx} min={0} max={3} step={0.05}
                fmt={(v) => (v ? v.toFixed(2) + "× floor" : "off")} on={(v) => setP("minPx", v)} />
              <div className="seg">
                {[["round", "Round"], ["square", "Square"]].map(([k, l]) => (
                  <button key={k} className={pc.shape === k ? "on" : ""} onClick={() => setP("shape", k)}>{l}</button>
                ))}
              </div>
              <Slide label="Opacity" v={pc.opacity} min={0.05} max={1} step={0.01} on={(v) => setP("opacity", v)} />
              <Toggle label="Perspective sizing" on={pc.atten} set={(v) => setP("atten", v)} />
              <Toggle label="Additive glow" on={pc.additive} set={(v) => setP("additive", v)} />
              <p className="hint">
                {mediaPts
                  ? "Anti-flicker sets a floor on how small a dot may get on screen — sub-pixel dots twinkle badly once they move."
                  : "Points are scattered by triangle area, not placed on vertices, so density stays even across the surface no matter how the mesh was built."}
              </p>
            </div>

            <div className="grp">
              <h3>Colour</h3>
              {!mediaPts && (
              <div className="seg">
                {[["texture", "Texture"], ["normal", "Normal"], ["height", "Height"], ["solid", "Solid"]].map(([k, l]) => (
                  <button key={k} className={pc.colorSrc === k ? "on" : ""} onClick={() => setP("colorSrc", k)}>{l}</button>
                ))}
              </div>
              )}
              <div className="swatches">
                {pc.colorSrc === "height" ? (
                  <>
                    <Swatch label="Low" value={pc.gradA} on={(v) => setP("gradA", v)} />
                    <Swatch label="High" value={pc.gradB} on={(v) => setP("gradB", v)} />
                  </>
                ) : (
                  <>
                    <Swatch label="Points" value={pc.solid} on={(v) => setP("solid", v)} />
                    <Swatch label="Field" value={pc.bg} on={(v) => setP("bg", v)} />
                  </>
                )}
              </div>
              {pc.colorSrc === "height" && (
                <div className="swatches">
                  <Swatch label="Field" value={pc.bg} on={(v) => setP("bg", v)} />
                </div>
              )}
            </div>

            <div className="grp">
              <h3>Form</h3>
              <Slide label="Scatter" v={pc.scatter} min={0} max={mediaPts ? 0.4 : 1.5} step={0.005} on={(v) => setP("scatter", v)} />
              <Slide label="Scale" v={pc.scale} min={0.2} max={4} step={0.01} on={(v) => setP("scale", v)} />
              <Slide label="Height" v={pc.height} min={-2} max={2} step={0.01} on={(v) => setP("height", v)} />
              <Slide label="Distance" v={pc.zoom} min={1.4} max={14} step={0.05} on={(v) => setP("zoom", v)} />
              <Slide label="Spin" v={pc.spin} min={-2} max={2} step={0.01} fmt={(v) => v.toFixed(2) + " r/s"} on={(v) => setP("spin", v)} />
              <p className="hint">{mediaPts ? "Scatter jitters every point in all three axes — the picture dissolves into noise." : "Scatter pushes each point along its surface normal — the model dissolves outward."}</p>
            </div>

            <div className="grp">
              <h3>Export</h3>
              <div className="exports">
                <button className="btn" onClick={exportScenePNG}>PNG</button>
                <button className="btn" onClick={exportPLY}>PLY</button>
                <button className="btn wide" onClick={recordLoop} disabled={recording || seq.busy}>
                  {recording ? `Recording ${tl.duration}s…` : `Record ${tl.duration}s · ${exp.height === 2160 ? "4K" : exp.height + "p"} · ${exp.fps}fps`}
                </button>
                <button className="btn wide" onClick={() => setPc(PC_DEFAULTS)}>Reset point cloud</button>
                <button className="btn wide" onClick={() => exportSequence(24, true)} disabled={seq.busy || recording || liveSource}>
                  {seq.busy ? `Rendering ${seq.done}/${seq.total}` : liveSource ? "PNG sequence needs a still source" : `PNG sequence · alpha · ${exp.fps}fps`}
                </button>
              </div>
              <p className="hint">PLY opens in Blender, CloudCompare, MeshLab, and Houdini.</p>
            </div>
          </div>
        )}

        {panelOpen && section === "collage" && (
          <div className="rail">
            <div className="grp">
              <h3>Source</h3>
              <div className="seg">
                {[["model", "Model"], ["image", "Image"], ["camera", "Camera"]].map(([k, l]) => (
                  <button key={k} className={src.kind === k ? "on" : ""} onClick={() => pickSource(k)}>{l}</button>
                ))}
              </div>
              <button className="btn" style={{ width: "100%" }} onClick={() => mediaInputRef.current.click()}>
                Open photo or video
              </button>
              <p className="hint">Camera is the one to try here — every panel treats the same face differently.</p>
            </div>

            <div className="grp">
              <h3>Styles</h3>
              <div className="chips">
                {[["cyberpunk", "Cyberpunk"], ["y2k", "Y2K"], ["brutalist", "Brutalist"], ["editorial", "Editorial"], ["glitch", "Glitch"], ["japanese", "Japanese Tech"], ["retro", "Retro Computer"], ["fashion", "Fashion"], ["blueprint", "Blueprint"], ["scrapbook", "Scrapbook"], ["terminal", "Terminal"], ["album", "Album Cover"], ["ransom", "Ransom Note"], ["vaporwave", "Vaporwave"]].map(([k, l]) => (
                  <button key={k} className="chip" onClick={() => applyCollage(k)}>{l}</button>
                ))}
              </div>
              <p className="hint">
                Each one sets the whole sheet at once — layout, framing, palette, trim and grain.
                Everything below stays editable afterwards.
              </p>
            </div>

            <div className="grp">
              <h3>Collage type</h3>
              <div className="chips">
                {[["grid", "Grid"], ["hero", "Hero"], ["strip", "Strip"], ["split", "Split"],
                  ["polaroid", "Polaroids"], ["mirror", "Mirror"], ["kaleido", "Kaleidoscope"]].map(([k, l]) => (
                  <button key={k} className={"chip" + (cl.layout === k ? " on" : "")} onClick={() => setC("layout", k)}>{l}</button>
                ))}
              </div>
              {cl.layout === "kaleido" ? (
                <Slide label="Wedges" v={cl.wedges} min={3} max={16} step={1} fmt={(v) => v + ""} on={(v) => setC("wedges", v)} />
              ) : cl.layout === "strip" ? (
                <Slide label="Frames" v={cl.rows} min={2} max={6} step={1} fmt={(v) => v + ""} on={(v) => setC("rows", v)} />
              ) : cl.layout === "split" ? (
                <Slide label="Panels" v={cl.cols} min={2} max={5} step={1} fmt={(v) => v + ""} on={(v) => setC("cols", v)} />
              ) : cl.layout !== "mirror" ? (
                <>
                  <div className="chips">
                    {[[1, 1, "Single"], [2, 1, "2 across"], [2, 2, "4"], [3, 2, "6"], [3, 3, "9"], [4, 3, "12"]].map(([c, r, l]) => (
                      <button key={l} className={"chip" + (cl.cols === c && cl.rows === r ? " on" : "")}
                        onClick={() => setCl((p) => ({ ...p, cols: c, rows: r }))}>{l}</button>
                    ))}
                  </div>
                  <Slide label="Columns" v={cl.cols} min={1} max={5} step={1} fmt={(v) => v + ""} on={(v) => setC("cols", v)} />
                  <Slide label="Rows" v={cl.rows} min={1} max={5} step={1} fmt={(v) => v + ""} on={(v) => setC("rows", v)} />
                </>
              ) : null}
              <p className="hint">
                {cl.layout === "grid" ? "Single gives one panel — the whole frame in one treatment."
                  : cl.layout === "mirror" ? "One frame reflected into four quadrants."
                  : cl.layout === "kaleido" ? "One frame folded into wedges around the centre."
                  : "Panel count is set above."}
              </p>
              {cl.layout === "polaroid" && (
                <Slide label="Scatter" v={cl.jitter} min={0} max={1} step={0.01} on={(v) => setC("jitter", v)} />
              )}
            </div>

            <div className="grp">
              <h3>Framing</h3>
              <div className="seg">
                {[["same", "Whole"], ["tiles", "Tiles"], ["dolly", "Dolly"], ["echo", "Echo"]].map(([k, l]) => (
                  <button key={k} className={cl.frameMode === k ? "on" : ""} onClick={() => setC("frameMode", k)}>{l}</button>
                ))}
              </div>
              <div className="seg">
                {[["cover", "Fill"], ["contain", "Fit"]].map(([k, l]) => (
                  <button key={k} className={cl.fit === k ? "on" : ""} onClick={() => setC("fit", k)}>{l}</button>
                ))}
              </div>
              <Slide label="Zoom" v={cl.zoom} min={0.15} max={6} step={0.01} fmt={(v) => v.toFixed(2) + "×"} on={(v) => setC("zoom", v)} />
              <button className="btn" style={{ width: "100%" }}
                onClick={() => setCl((p) => ({ ...p, zoom: 1, panX: 0, panY: 0 }))}>
                Reset view
              </button>
              <p className="hint">
                {cl.frameMode === "tiles" ? "The picture is cut across the panels — one image, reassembled."
                 : cl.frameMode === "dolly" ? "Each panel pushes further in, so the sheet reads as a zoom."
                 : cl.frameMode === "echo" ? "Each panel lags a little further behind — live motion leaves a trail."
                 : "Every panel shows the whole frame."}
                {" "}Scroll on the canvas to zoom, drag to pan.
              </p>
            </div>

            <div className="grp">
              <h3>Looks</h3>
              <div className="seg">
                {[["cycle", "One per panel"], ["same", "All the same"]].map(([k, l]) => (
                  <button key={k} className={cl.lookMode === k ? "on" : ""} onClick={() => setC("lookMode", k)}>{l}</button>
                ))}
              </div>
              <div className="chips">
                {LOOKS.map((lk, i) => (
                  <button key={lk.name} className={"chip" + (cl.look === i ? " on" : "")} onClick={() => setC("look", i)}>{lk.name}</button>
                ))}
              </div>
              <Slide label="Detail" v={cl.detail} min={20} max={220} step={2} fmt={(v) => v + " cols"} on={(v) => setC("detail", v)} />
              <Slide label="Coverage" v={cl.cut} min={0} max={0.9} step={0.005} fmt={(v) => (v ? v.toFixed(3) : "all")} on={(v) => setC("cut", v)} />
              <Slide label="Contrast" v={cl.contrast} min={0.2} max={3} step={0.01} on={(v) => setC("contrast", v)} />
              <Slide label="Brightness" v={cl.brightness} min={-0.6} max={0.6} step={0.01} on={(v) => setC("brightness", v)} />
              <Slide label="Photo behind" v={cl.backdrop} min={0} max={1} step={0.01} on={(v) => setC("backdrop", v)} />
              <Toggle label="Duotone (Warhol)" on={cl.duotone} set={(v) => setC("duotone", v)} />
              {cl.duotone && (
                <div className="chips" style={{ marginTop: 6 }}>
                  {DUOS.map((d, i) => (
                    <button key={i} className={"chip" + (cl.duoIdx === i ? " on" : "")}
                      onClick={() => setC("duoIdx", i)}
                      style={{ background: d[1], color: d[0], borderColor: cl.duoIdx === i ? d[0] : undefined }}>
                      {String(i + 1).padStart(2, "0")}
                    </button>
                  ))}
                </div>
              )}
              <Toggle label="Detail ramp" on={cl.ramp} set={(v) => setC("ramp", v)} />
              <Toggle label="Mirror" on={cl.mirror} set={(v) => setC("mirror", v)} />
              <p className="hint">
                {cl.duotone
                  ? "Duotone overrides colour: every panel gets its own two-ink palette."
                  : cl.lookMode === "same"
                  ? "Every panel gets the look you pick above."
                  : "Each panel takes the next look in the list, starting from the one you pick above."}
                {cl.ramp ? " Detail ramp coarsens the first panels and sharpens the last." : ""}
              </p>
            </div>

            <div className="grp">
              <h3>Photobooth</h3>
              <div className="seg">
                {[["live", "Live"], ["time", "Captured"]].map(([k, l]) => (
                  <button key={k} className={cl.mode === k ? "on" : ""} onClick={() => setC("mode", k)}>{l}</button>
                ))}
              </div>
              <Slide label="Shots" v={cl.shots} min={2} max={12} step={1} fmt={(v) => v + ""} on={(v) => setC("shots", v)} />
              <Slide label="Interval" v={cl.interval} min={200} max={2000} step={50} fmt={(v) => (v / 1000).toFixed(2) + "s"} on={(v) => setC("interval", v)} />
              <button className="btn pri" style={{ width: "100%", marginBottom: 6 }}
                onClick={runPhotobooth} disabled={shoot.busy || !src.name}>
                {shoot.busy ? (shoot.count ? `${shoot.count}…` : `Shot ${shoot.have}/${cl.shots}`) : "Start countdown"}
              </button>
              <button className="btn" style={{ width: "100%" }} onClick={clearShots}>Clear captures</button>
              <p className="hint">
                Live shows the same instant in every panel. Captured spreads a burst of stills
                across them, so the sheet reads as a sequence.
              </p>
            </div>

            <div className="grp">
              <h3>Sheet</h3>
              <Slide label="Gap" v={cl.gap} min={0} max={60} step={1} fmt={(v) => v + "px"} on={(v) => setC("gap", v)} />
              <Slide label="Corner" v={cl.radius} min={0} max={40} step={1} fmt={(v) => v + "px"} on={(v) => setC("radius", v)} />
              <Slide label="Border" v={cl.borderW} min={0} max={30} step={1} fmt={(v) => (v ? v + "px" : "none")} on={(v) => setC("borderW", v)} />
              <Slide label="Shadow" v={cl.shadow} min={0} max={1} step={0.01} on={(v) => setC("shadow", v)} />
              <Slide label="Tilt" v={cl.rot} min={0} max={1} step={0.01} fmt={(v) => (v ? v.toFixed(2) : "square")} on={(v) => setC("rot", v)} />
              <div className="swatches">
                <Swatch label="Matte" value={cl.matte} on={(v) => setC("matte", v)} />
                <Swatch label="Border" value={cl.border} on={(v) => setC("border", v)} />
              </div>
              <Slide label="Vignette" v={cl.vignette} min={0} max={1} step={0.01} on={(v) => setC("vignette", v)} />
              <Slide label="Film grain" v={cl.grain} min={0} max={0.6} step={0.01} on={(v) => setC("grain", v)} />
              <Slide label="Scan lines" v={cl.scan} min={0} max={0.8} step={0.01} on={(v) => setC("scan", v)} />
            </div>

            <div className="grp">
              <h3>Export</h3>
              <div className="exports">
                <button className="btn wide" onClick={exportScenePNG}>PNG</button>
                <button className="btn wide" onClick={recordLoop} disabled={recording || seq.busy}>
                  {recording ? `Recording ${tl.duration}s…` : `Record ${tl.duration}s · ${exp.height === 2160 ? "4K" : exp.height + "p"}`}
                </button>
                <button className="btn wide" onClick={() => setCl(CL_DEFAULTS)}>Reset collage</button>
              </div>
              <p className="hint">A 3×3 grid on camera with one look per panel is the shot worth posting.</p>
            </div>
          </div>
        )}

        {panelOpen && section === "media" && (
          <div className="rail">
            <div className="grp">
              <h3>Recipes</h3>
              <div className="chips">
                {Object.keys(RECIPES).map((k) => (
                  <button key={k} className="chip" onClick={() => applyRecipe(k)}>{k}</button>
                ))}
              </div>
              <button className="btn" style={{ width: "100%" }} onClick={() => mediaInputRef.current.click()}>
                Open photo or video
              </button>
              <p className="hint">
                Drop a file on the canvas or paste one from the clipboard. Nothing uploads —
                decoding and rendering both happen in this tab.
              </p>
            </div>

            <div className="grp">
              <h3>Style</h3>
              <div className="chips">
                {STYLES.map(([k, l]) => (
                  <button key={k} className={"chip" + (md.style === k ? " on" : "")} onClick={() => setM("style", k)}>{l}</button>
                ))}
              </div>
              {(md.style === "chars" || md.style === "mixed") && (
                <>
                  <div className="ramp">
                    <div className="strip" style={{ background: md.bg, color: md.tint }}>
                      {(md.ramp === "custom" ? md.custom : RAMPS[md.ramp]) || RAMPS.standard}
                    </div>
                    <input
                      value={md.ramp === "custom" ? md.custom : (RAMPS[md.ramp] || RAMPS.standard)}
                      onFocus={() => setMd((p) => ({ ...p, ramp: "custom", custom: p.custom || RAMPS[p.ramp] || RAMPS.standard }))}
                      onChange={(e) => setMd((p) => ({ ...p, ramp: "custom", custom: e.target.value }))}
                      spellCheck={false}
                      aria-label="Character ramp"
                    />
                  </div>
                  <div className="chips">
                    {Object.keys(RAMPS).map((k) => (
                      <button key={k} className={"chip" + (md.ramp === k ? " on" : "")} onClick={() => setM("ramp", k)}>{k}</button>
                    ))}
                  </div>
                </>
              )}
              {md.style === "dots" && (
                <>
                  <div className="seg">
                    {[["round", "Round"], ["square", "Square"], ["ring", "Ring"]].map(([k, l]) => (
                      <button key={k} className={md.shape === k ? "on" : ""} onClick={() => setM("shape", k)}>{l}</button>
                    ))}
                  </div>
                  <Slide label="Dot size" v={md.dot} min={0.2} max={1.8} step={0.01} on={(v) => setM("dot", v)} />
                </>
              )}
              {md.style === "voxel" && (
                <Slide label="Cube height" v={md.depth} min={0.05} max={2} step={0.01} on={(v) => setM("depth", v)} />
              )}
              <Slide label="Detail" v={md.detail} min={30} max={360} step={2} fmt={(v) => v + " cols"} on={(v) => setM("detail", v)} />
              <p className="hint">
                Braille packs 2×4 sub-samples into one glyph, so it resolves eight times the
                detail of a character cell at the same size.
              </p>
            </div>

            <div className="grp">
              <h3>Tone</h3>
              <Slide label="Coverage" v={md.cut} min={0} max={0.9} step={0.005} fmt={(v) => (v ? v.toFixed(3) : "all")} on={(v) => setM("cut", v)} />
              <Slide label="Contrast" v={md.contrast} min={0.2} max={3} step={0.01} on={(v) => setM("contrast", v)} />
              <Slide label="Brightness" v={md.brightness} min={-0.6} max={0.6} step={0.01} on={(v) => setM("brightness", v)} />
              <Slide label="Gamma" v={md.gamma} min={0.3} max={3} step={0.01} on={(v) => setM("gamma", v)} />
              <Toggle label="Invert" on={md.invert} set={(v) => setM("invert", v)} />
              <Toggle label="Mirror" on={md.mirror} set={(v) => setM("mirror", v)} />
              <div className="seg" style={{ marginTop: 10 }}>
                {[["cover", "Fill"], ["contain", "Fit"]].map(([k, l]) => (
                  <button key={k} className={md.fit === k ? "on" : ""} onClick={() => setM("fit", k)}>{l}</button>
                ))}
              </div>
              <Slide label="Zoom" v={md.zoom} min={0.15} max={8} step={0.01} fmt={(v) => v.toFixed(2) + "×"} on={(v) => setM("zoom", v)} />
              <button className="btn" style={{ width: "100%" }}
                onClick={() => setMd((p) => ({ ...p, zoom: 1, panX: 0, panY: 0 }))}>
                Reset view
              </button>
              <p className="hint">Scroll on the canvas to zoom, drag to pan.</p>
            </div>

            <div className="grp">
              <h3>Colour</h3>
              <div className="seg">
                {[["source", "Source"], ["mono", "Mono"]].map(([k, l]) => (
                  <button key={k} className={md.color === k ? "on" : ""} onClick={() => setM("color", k)}>{l}</button>
                ))}
              </div>
              <div className="chips">
                {Object.keys(GRADES).map((k) => (
                  <button key={k} className={"chip" + (md.grade === k ? " on" : "")} onClick={() => setM("grade", k)}>{k}</button>
                ))}
              </div>
              <Slide label="Saturation" v={md.sat} min={0} max={3} step={0.01} on={(v) => setM("sat", v)} />
              <div className="swatches">
                <Swatch label="Ink" value={md.tint} on={(v) => setM("tint", v)} />
                <Swatch label="Field" value={md.bg} on={(v) => setM("bg", v)} />
              </div>
            </div>

            <div className="grp">
              <h3>Backdrop</h3>
              <div className="seg">
                {[["none", "None"], ["orig", "Photo"], ["blur", "Blurred"]].map(([k, l]) => (
                  <button key={k} className={md.backdrop === k ? "on" : ""} onClick={() => setM("backdrop", k)}>{l}</button>
                ))}
              </div>
              {md.backdrop !== "none" && (
                <>
                  <Slide label="Opacity" v={md.backOpacity} min={0} max={1} step={0.01} on={(v) => setM("backOpacity", v)} />
                  {md.backdrop === "blur" && (
                    <Slide label="Blur" v={md.blur} min={0} max={60} step={1} fmt={(v) => v + "px"} on={(v) => setM("blur", v)} />
                  )}
                </>
              )}
              <p className="hint">
                Keeping the source behind the marks is what stops the result reading as flat —
                the art carries the detail, the backdrop carries the colour and depth.
              </p>
            </div>

            <div className="grp">
              <h3>Character FX</h3>
              <Slide label="Character bloom" v={md.cBloom} min={0} max={1} step={0.01} on={(v) => setM("cBloom", v)} />
              <Slide label="Character chromatic" v={md.cChroma} min={0} max={1} step={0.01} on={(v) => setM("cChroma", v)} />
              <p className="hint">
                These hit the mark layer alone, before it meets the backdrop — glowing glyphs
                over an untouched photo.
              </p>
            </div>

            <div className="grp">
              <h3>Frame FX</h3>
              <Slide label="Bloom" v={md.bloom} min={0} max={1} step={0.01} on={(v) => setM("bloom", v)} />
              <Slide label="Chromatic aberration" v={md.aberr} min={0} max={1} step={0.01} on={(v) => setM("aberr", v)} />
              <Slide label="CRT curvature" v={md.crt} min={0} max={1} step={0.01} on={(v) => setM("crt", v)} />
              <Slide label="Glitch" v={md.glitch} min={0} max={1} step={0.01} on={(v) => setM("glitch", v)} />
              <Slide label="Pixelate" v={md.pixel} min={0} max={1} step={0.01} on={(v) => setM("pixel", v)} />
              <Slide label="Halftone" v={md.halftone} min={0} max={1} step={0.01} on={(v) => setM("halftone", v)} />
              <Slide label="Scan lines" v={md.scan} min={0} max={0.8} step={0.01} on={(v) => setM("scan", v)} />
              <Slide label="Vignette" v={md.vignette} min={0} max={1} step={0.01} on={(v) => setM("vignette", v)} />
              <Slide label="Film grain" v={md.grain} min={0} max={0.6} step={0.01} on={(v) => setM("grain", v)} />
              <Slide label="Film dust" v={md.dust} min={0} max={1} step={0.01} on={(v) => setM("dust", v)} />
              <button className="btn" style={{ width: "100%", marginTop: 8 }}
                onClick={() => setMd((p) => ({ ...p, cBloom: 0, cChroma: 0, bloom: 0, aberr: 0, crt: 0, glitch: 0, pixel: 0, halftone: 0, scan: 0, vignette: 0, grain: 0, dust: 0 }))}>
                Clear all effects
              </button>
              <p className="hint">
                Applied in order: pixelate, halftone, aberration, bloom, glitch, scan lines,
                curvature, vignette, grain, dust.
              </p>
            </div>

            <div className="grp">
              <h3>Export</h3>
              <div className="exports">
                <button className="btn" onClick={exportScenePNG}>PNG</button>
                <button className="btn" onClick={copyText} disabled={!GLYPH_STYLES.has(md.style)}>Copy</button>
                <button className="btn wide" onClick={recordLoop} disabled={recording || seq.busy}>
                  {recording ? `Recording ${tl.duration}s…` : `Record ${tl.duration}s · ${exp.height === 2160 ? "4K" : exp.height + "p"}`}
                </button>
                <button className="btn wide" onClick={() => setMd(MD_DEFAULTS)}>Reset media</button>
              </div>
              <p className="hint">Resolution and frame rate come from the timeline bar below.</p>
            </div>
          </div>
        )}

        {panelOpen && section === "dither" && (
          <div className="rail">
            {mediaOn && (
              <div className="grp">
                <h3>Source</h3>
                <div className="seg">
                  {[["model", "Model"], ["image", "Image"], ["camera", "Camera"]].map(([k, l]) => (
                    <button key={k} className={src.kind === k ? "on" : ""} onClick={() => pickSource(k)}>{l}</button>
                  ))}
                </div>
                <div className="seg">
                  {[["cover", "Fill"], ["contain", "Fit"]].map(([k, l]) => (
                    <button key={k} className={src.fit === k ? "on" : ""} onClick={() => setSrc((p) => ({ ...p, fit: k }))}>{l}</button>
                  ))}
                </div>
                <Slide label="Cutout" v={src.cut} min={0} max={0.9} step={0.005}
                  fmt={(v) => (v ? v.toFixed(3) : "off")} on={(v) => setSrc((p) => ({ ...p, cut: v }))} />
                <Toggle label="Mirror" on={src.mirror} set={(v) => setSrc((p) => ({ ...p, mirror: v }))} />
                {src.kind === "image" && (
                  <button className="btn" style={{ width: "100%", marginTop: 8 }} onClick={() => imgInputRef.current.click()}>
                    Open another image
                  </button>
                )}
                <p className="hint">
                  Cutout decides which cells count as empty. A camera frame is opaque everywhere,
                  so raise it until the background drops away and only your subject is drawn.
                </p>
              </div>
            )}
            <div className="grp">
              <h3>Dither</h3>
              <div className="seg">
                {[["fs", "Diffusion"], ["bayer", "Ordered"], ["none", "None"]].map(([k, l]) => (
                  <button key={k} className={dt.algo === k ? "on" : ""} onClick={() => setD("algo", k)}>{l}</button>
                ))}
              </div>
              {dt.algo === "bayer" && (
                <div className="seg">
                  {[2, 4, 8].map((k) => (
                    <button key={k} className={dt.bayer === k ? "on" : ""} onClick={() => setD("bayer", k)}>{k}×{k}</button>
                  ))}
                </div>
              )}
              <Slide label="Resolution" v={dt.res} min={32} max={420} step={2} fmt={(v) => v + " px"} on={(v) => setD("res", v)} />
              {dt.algo !== "none" && (
                <Slide label="Strength" v={dt.spread} min={0} max={1.6} step={0.01} on={(v) => setD("spread", v)} />
              )}
              <p className="hint">
                Diffusion pushes each pixel's rounding error into its neighbours — organic grain.
                Ordered uses a fixed threshold grid — the regular crosshatch of old print and games.
              </p>
            </div>

            <div className="grp">
              <h3>Palette</h3>
              <div className="chips">
                {["mono", "gameboy", "cga", "zx", "teletext", "posterise"].map((k) => (
                  <button key={k} className={"chip" + (dt.palette === k ? " on" : "")} onClick={() => setD("palette", k)}>{k}</button>
                ))}
              </div>
              {dt.palette === "posterise" && (
                <Slide label="Levels" v={dt.levels} min={2} max={8} step={1} fmt={(v) => v + " / channel"} on={(v) => setD("levels", v)} />
              )}
              {dt.palette === "mono" && (
                <div className="swatches">
                  <Swatch label="Ink" value={dt.ink} on={(v) => setD("ink", v)} />
                  <Swatch label="Field" value={dt.bg} on={(v) => setD("bg", v)} />
                </div>
              )}
              {dt.palette !== "mono" && (
                <div className="swatches">
                  <Swatch label="Field" value={dt.bg} on={(v) => setD("bg", v)} />
                </div>
              )}
            </div>

            <div className="grp">
              <h3>Tone</h3>
              <Slide label="Contrast" v={dt.contrast} min={0.2} max={3} step={0.01} on={(v) => setD("contrast", v)} />
              <Slide label="Brightness" v={dt.brightness} min={-0.6} max={0.6} step={0.01} on={(v) => setD("brightness", v)} />
              <Slide label="Gamma" v={dt.gamma} min={0.3} max={3} step={0.01} on={(v) => setD("gamma", v)} />
              {!mediaOn && (
                <>
                  <Slide label="Ambient" v={dt.ambient} min={0} max={1.2} step={0.01} on={(v) => setD("ambient", v)} />
                  <Slide label="Key" v={dt.key} min={0} max={3} step={0.01} on={(v) => setD("key", v)} />
                </>
              )}
              <Toggle label="Invert" on={dt.invert} set={(v) => setD("invert", v)} />
            </div>

            {!mediaOn && (
            <div className="grp">
              <h3>Form</h3>
              <Slide label="Scale" v={dt.scale} min={0.2} max={4} step={0.01} on={(v) => setD("scale", v)} />
              <Slide label="Height" v={dt.height} min={-2} max={2} step={0.01} on={(v) => setD("height", v)} />
              <Slide label="Distance" v={dt.zoom} min={1.4} max={14} step={0.05} on={(v) => setD("zoom", v)} />
              <Slide label="Spin" v={dt.spin} min={-2} max={2} step={0.01} fmt={(v) => v.toFixed(2) + " r/s"} on={(v) => setD("spin", v)} />
            </div>
            )}

            <div className="grp">
              <h3>Export</h3>
              <div className="exports">
                <button className="btn" onClick={() => exportDitherPNG(false)}>PNG</button>
                <button className="btn" onClick={() => exportDitherPNG(true)}>PNG 1:1</button>
                <button className="btn wide" onClick={recordLoop} disabled={recording || seq.busy}>
                  {recording ? `Recording ${tl.duration}s…` : `Record ${tl.duration}s · ${exp.height === 2160 ? "4K" : exp.height + "p"} · ${exp.fps}fps`}
                </button>
                <button className="btn wide" onClick={() => setDt(DT_DEFAULTS)}>Reset dither</button>
                <button className="btn wide" onClick={() => exportSequence(24, true)} disabled={seq.busy || recording || liveSource}>
                  {seq.busy ? `Rendering ${seq.done}/${seq.total}` : liveSource ? "PNG sequence needs a still source" : `PNG sequence · alpha · ${exp.fps}fps`}
                </button>
              </div>
              <p className="hint">PNG 1:1 saves at the true pixel grid size. The sequence exports with a real alpha channel for After Effects.</p>
            </div>
          </div>
        )}

        {panelOpen && section === "voxel" && (
          <div className="rail">
            <HandPanel />
            <div className="grp">
              <h3>Grid</h3>
              <Slide label="Resolution" v={vx.res} min={12} max={140} step={1}
                fmt={(v) => v + "³"} on={(v) => setV("res", v)} />
              <Slide label="Gap" v={vx.gap} min={0} max={0.6} step={0.01} on={(v) => setV("gap", v)} />
              <Toggle label="Lit shading" on={vx.lit} set={(v) => setV("lit", v)} />
              {vx.lit && (
                <>
                  <Slide label="Ambient" v={vx.ambient} min={0} max={1.2} step={0.01} on={(v) => setV("ambient", v)} />
                  <Slide label="Key" v={vx.key} min={0} max={3} step={0.01} on={(v) => setV("key", v)} />
                </>
              )}
              <p className="hint">
                Voxels come from dense surface sampling, so concave and open meshes voxelise
                cleanly instead of collapsing to their bounding box.
              </p>
            </div>

            <div className="grp">
              <h3>Colour</h3>
              <div className="seg">
                {[["texture", "Texture"], ["normal", "Normal"], ["height", "Height"], ["solid", "Solid"]].map(([k, l]) => (
                  <button key={k} className={vx.colorSrc === k ? "on" : ""} onClick={() => setV("colorSrc", k)}>{l}</button>
                ))}
              </div>
              <div className="swatches">
                {vx.colorSrc === "height" ? (
                  <>
                    <Swatch label="Low" value={vx.gradA} on={(v) => setV("gradA", v)} />
                    <Swatch label="High" value={vx.gradB} on={(v) => setV("gradB", v)} />
                  </>
                ) : (
                  <>
                    <Swatch label="Blocks" value={vx.solid} on={(v) => setV("solid", v)} />
                    <Swatch label="Field" value={vx.bg} on={(v) => setV("bg", v)} />
                  </>
                )}
              </div>
            </div>

            <div className="grp">
              <h3>Form</h3>
              <Slide label="Scatter" v={vx.scatter} min={0} max={0.8} step={0.005} on={(v) => setV("scatter", v)} />
              <Slide label="Scale" v={vx.scale} min={0.2} max={4} step={0.01} on={(v) => setV("scale", v)} />
              <Slide label="Height" v={vx.height} min={-2} max={2} step={0.01} on={(v) => setV("height", v)} />
              <Slide label="Distance" v={vx.zoom} min={1.4} max={14} step={0.05} on={(v) => setV("zoom", v)} />
              <Slide label="Spin" v={vx.spin} min={-2} max={2} step={0.01} fmt={(v) => v.toFixed(2) + " r/s"} on={(v) => setV("spin", v)} />
              <p className="hint">Scatter pushes each block outward along the surface it was sampled from.</p>
            </div>

            <div className="grp">
              <h3>Export</h3>
              <div className="exports">
                <button className="btn" onClick={exportScenePNG}>PNG</button>
                <button className="btn" onClick={exportVoxelCSV}>CSV</button>
                <button className="btn wide" onClick={recordLoop} disabled={recording || seq.busy}>
                  {recording ? `Recording ${tl.duration}s…` : `Record ${tl.duration}s · ${exp.height === 2160 ? "4K" : exp.height + "p"} · ${exp.fps}fps`}
                </button>
                <button className="btn wide" onClick={() => setVx(VX_DEFAULTS)}>Reset voxels</button>
                <button className="btn wide" onClick={() => exportSequence(24, true)} disabled={seq.busy || recording || liveSource}>
                  {seq.busy ? `Rendering ${seq.done}/${seq.total}` : liveSource ? "PNG sequence needs a still source" : `PNG sequence · alpha · ${exp.fps}fps`}
                </button>
              </div>
              <p className="hint">CSV gives integer grid coordinates and a hex colour per block.</p>
            </div>
          </div>
        )}

        {panelOpen && section === "wire" && (
          <div className="rail">
            <HandPanel />
            <div className="grp">
              <h3>Edges</h3>
              <div className="seg">
                {[["edges", "Feature"], ["all", "Every triangle"]].map(([k, l]) => (
                  <button key={k} className={wf.mode === k ? "on" : ""} onClick={() => setW("mode", k)}>{l}</button>
                ))}
              </div>
              {wf.mode === "edges" && (
                <Slide label="Crease angle" v={wf.angle} min={1} max={80} step={1}
                  fmt={(v) => v + "°"} on={(v) => setW("angle", v)} />
              )}
              <Slide label="Opacity" v={wf.opacity} min={0.05} max={1} step={0.01} on={(v) => setW("opacity", v)} />
              <Toggle label="Hide back edges" on={wf.hideBack} set={(v) => setW("hideBack", v)} />
              <Toggle label="Additive glow" on={wf.additive} set={(v) => setW("additive", v)} />
              <p className="hint">
                Feature mode keeps only edges where two faces meet above the crease angle. Raise it
                to strip triangulation noise, lower it to expose the mesh topology.
              </p>
            </div>

            <div className="grp">
              <h3>Depth</h3>
              <Slide label="Fog" v={wf.fog} min={0} max={1} step={0.01} on={(v) => setW("fog", v)} />
              <div className="swatches">
                <Swatch label="Line" value={wf.color} on={(v) => setW("color", v)} />
                <Swatch label="Field" value={wf.bg} on={(v) => setW("bg", v)} />
              </div>
              <p className="hint">Fog fades distant edges toward the field colour, which reads as depth.</p>
            </div>

            <div className="grp">
              <h3>Form</h3>
              <Slide label="Scale" v={wf.scale} min={0.2} max={4} step={0.01} on={(v) => setW("scale", v)} />
              <Slide label="Height" v={wf.height} min={-2} max={2} step={0.01} on={(v) => setW("height", v)} />
              <Slide label="Distance" v={wf.zoom} min={1.4} max={14} step={0.05} on={(v) => setW("zoom", v)} />
              <Slide label="Spin" v={wf.spin} min={-2} max={2} step={0.01} fmt={(v) => v.toFixed(2) + " r/s"} on={(v) => setW("spin", v)} />
            </div>

            <div className="grp">
              <h3>Plotter</h3>
              <div className="chips">
                {Object.keys(PAPERS).map((k) => (
                  <button key={k} className={"chip" + (plot.paper === k ? " on" : "")} onClick={() => setPl("paper", k)}>{k}</button>
                ))}
              </div>
              <Toggle label="Landscape" on={plot.landscape} set={(v) => setPl("landscape", v)} />
              <Toggle label="Hidden line removal" on={plot.hlr} set={(v) => setPl("hlr", v)} />
              <Toggle label="Hidden edges (dashed reference layer)" on={plot.hidden} set={(v) => setPl("hidden", v)} />
              {plot.hidden && (
                <p className="hint" style={{ margin: "2px 0 10px", color: "var(--sulfur)" }}>
                  Occluded edges are most of the model. Keep this off unless you specifically want
                  a drafting-style reference layer to plot in a second pen.
                </p>
              )}
              <Slide label="Margin" v={plot.margin} min={0} max={40} step={1} fmt={(v) => v + " mm"} on={(v) => setPl("margin", v)} />
              <Slide label="Pen width" v={plot.pen} min={0.05} max={2} step={0.05} fmt={(v) => v.toFixed(2) + " mm"} on={(v) => setPl("pen", v)} />
              <Slide label="Accuracy" v={plot.quality} min={500} max={2400} step={50} fmt={(v) => v + " px"} on={(v) => setPl("quality", v)} />
              <Slide label="Tolerance" v={plot.tolerance} min={1} max={40} step={1} fmt={(v) => v.toFixed(0)} on={(v) => setPl("tolerance", v)} />
              <Slide label="Drop specks under" v={plot.minLen} min={0} max={4} step={0.05}
                fmt={(v) => (v ? v.toFixed(2) + " mm" : "keep all")} on={(v) => setPl("minLen", v)} />
              <Toggle label="Fit drawing to page" on={plot.fit} set={(v) => setPl("fit", v)} />
              <Toggle label="Join touching lines" on={plot.join} set={(v) => setPl("join", v)} />
              <button className="btn pri" style={{ width: "100%", marginTop: 8 }} onClick={exportWireSVG} disabled={busy}>
                {busy ? "Working\u2026" : "Export plotter SVG"}
              </button>
              {plotInfo && (
                <div style={{
                  marginTop: 10, padding: "9px 11px", borderRadius: 8,
                  background: "rgba(255,255,255,.04)", border: "1px solid var(--line2)",
                  fontFamily: MONO, fontSize: 11, lineHeight: 1.7, color: "var(--dim)",
                }}>
                  <div><span style={{ color: "var(--text)" }}>{plotInfo.paths.toLocaleString()}</span> pen strokes from {plotInfo.edges.toLocaleString()} edges</div>
                  <div><span style={{ color: "var(--text)" }}>{(plotInfo.ink / 1000).toFixed(2)} m</span> of ink</div>
                  <div>about <span style={{ color: "var(--text)" }}>{plotInfo.minutes < 1 ? "<1" : Math.round(plotInfo.minutes)} min</span> to plot \u00b7 {plotInfo.ms}ms to compute</div>
                </div>
              )}
              <p className="hint">
                A plotter cannot paint faces over lines, so edges behind the form are deleted
                before export. If the drawing comes out as a solid black mass there are simply too
                many edges \u2014 raise the <b>crease angle</b> above until only real panel lines
                remain. Joining lines keeps the pen down and cuts plotting time many times over.
                Raise <b>tolerance</b> if edges on a surface flicker out, lower it if hidden lines
                leak through.
              </p>
            </div>

            <div className="grp">
              <h3>Export</h3>
              <div className="exports">
                <button className="btn" onClick={exportScenePNG}>PNG</button>
                <button className="btn" onClick={exportWireSVG}>SVG</button>
                <button className="btn wide" onClick={recordLoop} disabled={recording || seq.busy}>
                  {recording ? `Recording ${tl.duration}s…` : `Record ${tl.duration}s · ${exp.height === 2160 ? "4K" : exp.height + "p"} · ${exp.fps}fps`}
                </button>
                <button className="btn wide" onClick={() => setWf(WF_DEFAULTS)}>Reset wireframe</button>
                <button className="btn wide" onClick={() => exportSequence(24, true)} disabled={seq.busy || recording || liveSource}>
                  {seq.busy ? `Rendering ${seq.done}/${seq.total}` : liveSource ? "PNG sequence needs a still source" : `PNG sequence · alpha · ${exp.fps}fps`}
                </button>
              </div>
              <p className="hint">SVG exports the current view as real vector lines, ready for a plotter.</p>
            </div>
          </div>
        )}

        {panelOpen && section === "ascii" && (
          <div className="rail">
            <HandPanel />
            {mediaOn && (
              <div className="grp">
                <h3>Source</h3>
                <div className="seg">
                  {[["model", "Model"], ["image", "Image"], ["camera", "Camera"]].map(([k, l]) => (
                    <button key={k} className={src.kind === k ? "on" : ""} onClick={() => pickSource(k)}>{l}</button>
                  ))}
                </div>
                <div className="seg">
                  {[["cover", "Fill"], ["contain", "Fit"]].map(([k, l]) => (
                    <button key={k} className={src.fit === k ? "on" : ""} onClick={() => setSrc((p) => ({ ...p, fit: k }))}>{l}</button>
                  ))}
                </div>
                <Slide label="Cutout" v={src.cut} min={0} max={0.9} step={0.005}
                  fmt={(v) => (v ? v.toFixed(3) : "off")} on={(v) => setSrc((p) => ({ ...p, cut: v }))} />
                <Toggle label="Mirror" on={src.mirror} set={(v) => setSrc((p) => ({ ...p, mirror: v }))} />
                {src.kind === "image" && (
                  <button className="btn" style={{ width: "100%", marginTop: 8 }} onClick={() => imgInputRef.current.click()}>
                    Open another image
                  </button>
                )}
                <p className="hint">
                  Cutout decides which cells count as empty. A camera frame is opaque everywhere,
                  so raise it until the background drops away and only your subject is drawn.
                </p>
              </div>
            )}
            <div className="grp">
              <h3>Mark</h3>
              <div className="seg">
                {[["dots", "Dots"], ["glyphs", "Glyphs"]].map(([k, l]) => (
                  <button key={k} className={s.mark === k ? "on" : ""} onClick={() => set("mark", k)}>{l}</button>
                ))}
              </div>

              {s.mark === "dots" ? (
                <>
                  <div className="seg">
                    {[["round", "Round"], ["square", "Square"], ["ring", "Ring"]].map(([k, l]) => (
                      <button key={k} className={s.shape === k ? "on" : ""} onClick={() => set("shape", k)}>{l}</button>
                    ))}
                  </div>
                  <Slide label="Dot size" v={s.dot} min={0.2} max={1.8} step={0.01} on={(v) => set("dot", v)} />
                </>
              ) : (
                <>
              <div className="ramp">
                <div className="strip" style={{ background: s.bg, color: s.tint }}>{ramp}</div>
                <input
                  value={s.ramp === "custom" ? s.custom : ramp}
                  onFocus={() => setS((p) => ({ ...p, ramp: "custom", custom: p.custom || ramp }))}
                  onChange={(e) => setS((p) => ({ ...p, ramp: "custom", custom: e.target.value }))}
                  spellCheck={false}
                  aria-label="Character ramp, darkest first"
                />
              </div>
              <div className="chips">
                {Object.keys(RAMPS).map((k) => (
                  <button key={k} className={"chip" + (s.ramp === k ? " on" : "")} onClick={() => set("ramp", k)}>
                    {k}
                  </button>
                ))}
              </div>
                </>
              )}
              <Slide label="Detail" v={s.detail} min={24} max={240} step={1} fmt={(v) => v + " cols"} on={(v) => set("detail", v)} />
              <Slide label="Contrast" v={s.contrast} min={0.2} max={3} step={0.01} on={(v) => set("contrast", v)} />
              <Slide label="Brightness" v={s.brightness} min={-0.6} max={0.6} step={0.01} on={(v) => set("brightness", v)} />
              <Slide label="Gamma" v={s.gamma} min={0.3} max={3} step={0.01} on={(v) => set("gamma", v)} />
              <Slide label="Cell height" v={s.lineHeight} min={0.7} max={1.6} step={0.01} on={(v) => set("lineHeight", v)} />
              <Toggle label="Invert" on={s.invert} set={(v) => set("invert", v)} />
            </div>

            {!mediaOn && (
            <div className="grp">
              <h3>Source pass</h3>
              <div className="seg">
                {[["shaded", "Shaded"], ["depth", "Depth"], ["normal", "Normal"]].map(([k, l]) => (
                  <button key={k} className={s.pass === k ? "on" : ""} onClick={() => set("pass", k)}>{l}</button>
                ))}
              </div>
              <Toggle label="Edge glyphs" on={s.edges} set={(v) => set("edges", v)} />
              {s.edges && (
                <Slide label="Edge threshold" v={s.edgeAmount} min={0.02} max={0.8} step={0.01} on={(v) => set("edgeAmount", v)} />
              )}
            </div>
            )}

            <div className="grp">
              <h3>Colour</h3>
              <div className="seg">
                {[["mono", "Mono"], ["gradient", "Ramped"], ["model", "Model"]].map(([k, l]) => (
                  <button key={k} className={s.color === k ? "on" : ""} onClick={() => set("color", k)}>{l}</button>
                ))}
              </div>
              <div className="swatches">
                <Swatch label="Ink" value={s.tint} on={(v) => set("tint", v)} />
                <Swatch label="Field" value={s.bg} on={(v) => set("bg", v)} />
              </div>
              <div className="chips">
                {Object.keys(THEMES).map((k) => (
                  <button key={k} className="chip" onClick={() => applyTheme(k)}>{k}</button>
                ))}
              </div>
              {s.color === "model" && (
                <>
                  <Toggle label="Unlit colour" on={s.flat} set={(v) => set("flat", v)} />
                  <Slide label="Saturation" v={s.sat} min={0} max={3} step={0.01} on={(v) => set("sat", v)} />
                </>
              )}
              <Toggle label="Scanlines" on={s.scanlines} set={(v) => set("scanlines", v)} />
              <Toggle label="Glow" on={s.glow} set={(v) => set("glow", v)} />
            </div>

            {!mediaOn && (
            <div className="grp">
              <h3>Model</h3>
              <div className="chips">
                {["knot", "torus", "ico", "sphere", "cube"].map((k) => (
                  <button key={k} className="chip" onClick={() => { swap(primitive(k), k); setError(""); }}>
                    {k}
                  </button>
                ))}
              </div>
              <Slide label="Scale" v={s.scale} min={0.2} max={4} step={0.01} on={(v) => set("scale", v)} />
              <Slide label="Height" v={s.height} min={-2} max={2} step={0.01} on={(v) => set("height", v)} />
              <Slide label="Distance" v={s.zoom} min={1.4} max={14} step={0.05} on={(v) => set("zoom", v)} />
              <Slide label="Spin" v={s.spin} min={-2} max={2} step={0.01} fmt={(v) => v.toFixed(2) + " r/s"} on={(v) => set("spin", v)} />
              <Toggle label="Model colours" on={s.albedo} set={(v) => set("albedo", v)} />
              <Toggle label="Normal maps" on={s.bump} set={(v) => set("bump", v)} />
              <button className="btn" style={{ width: "100%", marginTop: 8 }} onClick={frameModel}>
                Fit to view
              </button>
              <p className="hint">
                Drag the canvas to orbit. Scroll to push the camera. Textures are read straight
                out of the .glb, so switch Colour to Model to see them in the glyphs.
              </p>
            </div>
            )}

            {!mediaOn && (
            <div className="grp">
              <h3>Light</h3>
              <Slide label="Ambient" v={s.ambient} min={0} max={1.5} step={0.01} on={(v) => set("ambient", v)} />
              <Slide label="Key" v={s.key} min={0} max={5} step={0.01} on={(v) => set("key", v)} />
              <Slide label="Fill" v={s.fill} min={0} max={3} step={0.01} on={(v) => set("fill", v)} />
              <Slide label="Key angle" v={s.keyAngle} min={0} max={360} step={1} fmt={(v) => v + "°"} on={(v) => set("keyAngle", v)} />
              <Slide label="Specular" v={s.spec} min={0} max={1} step={0.01} on={(v) => set("spec", v)} />
            </div>
            )}

            <div className="grp">
              <h3>Export</h3>
              <div className="exports">
                <button className="btn" onClick={exportPNG}>PNG</button>
                <button className="btn" onClick={exportSVG}>SVG</button>
                <button className="btn" onClick={exportTXT} disabled={s.mark === "dots"}>Text</button>
                <button className="btn" onClick={copyText} disabled={s.mark === "dots"}>Copy</button>
                <button className="btn wide" onClick={recordLoop} disabled={recording || seq.busy}>
                  {recording ? `Recording ${tl.duration}s…` : `Record ${tl.duration}s · ${exp.height === 2160 ? "4K" : exp.height + "p"} · ${exp.fps}fps`}
                </button>
                <button className="btn wide" onClick={copySettings}>Copy settings JSON</button>
                <button className="btn wide" onClick={() => setS(DEFAULTS)}>Reset everything</button>
                <button className="btn wide" onClick={() => exportSequence(24, true)} disabled={seq.busy || recording || liveSource}>
                  {seq.busy ? `Rendering ${seq.done}/${seq.total}` : liveSource ? "PNG sequence needs a still source" : `PNG sequence · alpha · ${exp.fps}fps`}
                </button>
              </div>
              <p className="hint">Files never leave your device. Parsing and rendering both run in this tab.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---- small controls ---- */

function Slide({ label, v, min, max, step, on, fmt }) {
  return (
    <div>
      <div className="row">
        <label>{label}</label>
        <span className="val">{fmt ? fmt(v) : Number(v).toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={v}
        aria-label={label}
        onChange={(e) => on(parseFloat(e.target.value))}
      />
    </div>
  );
}

function Toggle({ label, on, set }) {
  return (
    <div
      className={"tog" + (on ? " on" : "")}
      role="switch"
      aria-checked={on}
      tabIndex={0}
      onClick={() => set(!on)}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), set(!on))}
    >
      <span>{label}</span>
      <div className="sw"><i /></div>
    </div>
  );
}

function Swatch({ label, value, on }) {
  return (
    <div style={{ flex: 1 }}>
      <div className="row" style={{ marginBottom: 5 }}>
        <label>{label}</label>
        <span className="val">{value.toUpperCase()}</span>
      </div>
      <div className="sw2" style={{ background: value }}>
        <input type="color" value={value} aria-label={label} onChange={(e) => on(e.target.value)} />
      </div>
    </div>
  );
}

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}
