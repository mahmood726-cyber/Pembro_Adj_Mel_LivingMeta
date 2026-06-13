#!/usr/bin/env node
/*
 * Minimal smoke test for the Pembro_Adj_Mel_LivingMeta statistical engine.
 *
 * The shipped app (PEMBRO_ADJ_MEL_REVIEW.html) computes pooled effects inside
 * an inline "REML-RETROFIT" engine. That code is not importable as a module, so
 * this test re-implements the exact same formulas it uses and asserts them
 * against independently hand-computed reference values. If the engine's math is
 * ever changed in a way that diverges from these references, this test should be
 * updated in lockstep (and only after re-deriving the reference by hand).
 *
 * Run: node test/smoke.test.js
 */
'use strict';

var assert = require('assert');

// ---- formulas mirrored from PEMBRO_ADJ_MEL_REVIEW.html ----

// log-OR / log-RR with conditional 0.5 zero-cell correction
function binaryEffect(d, measure) {
  var tE = d.tE, tN = d.tN, cE = d.cE, cN = d.cN;
  if (tN <= 0 || cN <= 0) return null;
  var tNonE = tN - tE, cNonE = cN - cE;
  if (tNonE < 0 || cNonE < 0) return null;
  var add = (tE === 0 || tNonE === 0 || cE === 0 || cNonE === 0) ? 0.5 : 0;
  var a = tE + add, b = tNonE + add, c = cE + add, e = cNonE + add;
  var y, v;
  if (measure === 'RR') {
    var pt = a / (a + b), pc = c / (c + e);
    y = Math.log(pt / pc);
    v = (1 / a) - (1 / (a + b)) + (1 / c) - (1 / (c + e));
  } else {
    y = Math.log(a * e / (b * c));
    v = 1 / a + 1 / b + 1 / c + 1 / e;
  }
  return { y: y, v: v };
}

// log-HR + variance from a published HR and its 95% CI
function hrFromPublished(hr, lo, hi) {
  var y = Math.log(hr);
  var se = (Math.log(hi) - Math.log(lo)) / (2 * 1.959964);
  return { y: y, v: se * se };
}

// Paule-Mandel tau^2 (Newton-Raphson on Q - (k-1))
function pauleMandelTau2(y, v) {
  var k = y.length;
  if (k < 2) return 0;
  var tau2 = 0, target = k - 1;
  for (var iter = 0; iter < 200; iter++) {
    var w = v.map(function (vi) { return 1 / (vi + tau2); });
    var sumW = w.reduce(function (a, b) { return a + b; }, 0);
    if (!(sumW > 0)) break;
    var yBar = 0;
    for (var i = 0; i < k; i++) yBar += y[i] * w[i];
    yBar /= sumW;
    var Q = 0;
    for (var j = 0; j < k; j++) { var dd = y[j] - yBar; Q += w[j] * dd * dd; }
    var diff = Q - target;
    if (Math.abs(diff) < 1e-5) break;
    var slope = 0;
    for (var m = 0; m < k; m++) { var dm = y[m] - yBar; slope += -w[m] * w[m] * dm * dm; }
    if (!Number.isFinite(slope) || Math.abs(slope) < 1e-14) break;
    var t2new = Math.max(0, tau2 - diff / slope);
    if (Math.abs(t2new - tau2) < 1e-9) { tau2 = t2new; break; }
    tau2 = t2new;
  }
  return tau2;
}

// inverse-variance pooling on the (log) scale
function poolWith(y, v, tau2) {
  var z = 1.959964, k = y.length;
  var w = v.map(function (vi) { return 1 / (vi + tau2); });
  var sumW = w.reduce(function (a, b) { return a + b; }, 0);
  var effect = 0;
  for (var i = 0; i < k; i++) effect += y[i] * w[i];
  effect /= sumW;
  var se = 1 / Math.sqrt(sumW);
  return { effect: effect, se: se, lo: effect - z * se, hi: effect + z * se };
}

function approx(a, b, tol, msg) {
  assert.ok(Math.abs(a - b) <= tol, msg + ' (got ' + a + ', want ' + b + ')');
}

var passed = 0;
function ok(name, fn) { fn(); passed++; console.log('  ok - ' + name); }

console.log('Pembro_Adj_Mel_LivingMeta smoke test');

// 1. log-OR + variance, no zero cell. a=10,b=90,c=20,e=80
ok('binaryEffect OR (no zero cell)', function () {
  var e = binaryEffect({ tE: 10, tN: 100, cE: 20, cN: 100 }, 'OR');
  approx(e.y, Math.log((10 * 80) / (90 * 20)), 1e-12, 'logOR');
  approx(e.v, 1 / 10 + 1 / 90 + 1 / 20 + 1 / 80, 1e-12, 'var OR');
});

// 2. zero-cell correction applied only when a cell is zero
ok('binaryEffect zero-cell adds 0.5', function () {
  var e = binaryEffect({ tE: 0, tN: 100, cE: 20, cN: 100 }, 'OR');
  // a=0.5, b=100-0+0.5=100.5, c=20.5, e=80.5
  approx(e.y, Math.log((0.5 * 80.5) / (100.5 * 20.5)), 1e-12, 'logOR with cc');
  assert.ok(Number.isFinite(e.y) && Number.isFinite(e.v), 'finite with zero cell');
});

// 3. RR log scale
ok('binaryEffect RR', function () {
  var e = binaryEffect({ tE: 10, tN: 100, cE: 20, cN: 100 }, 'RR');
  approx(e.y, Math.log((10 / 100) / (20 / 100)), 1e-12, 'logRR');
});

// 4. log-HR from published HR + CI
ok('hrFromPublished log-HR + SE', function () {
  var e = hrFromPublished(0.65, 0.46, 0.92);
  approx(e.y, Math.log(0.65), 1e-12, 'logHR');
  var seExpected = (Math.log(0.92) - Math.log(0.46)) / (2 * 1.959964);
  approx(e.v, seExpected * seExpected, 1e-12, 'var HR');
});

// 5. Paule-Mandel: identical effects -> tau2 = 0 (Q < k-1 floors at 0)
ok('pauleMandelTau2 homogeneous -> 0', function () {
  var t = pauleMandelTau2([0.1, 0.1, 0.1], [0.04, 0.04, 0.04]);
  approx(t, 0, 1e-9, 'tau2 should floor at 0');
});

// 6. Paule-Mandel: heterogeneous data -> tau2 > 0 and Q(tau2) ~= k-1
ok('pauleMandelTau2 heterogeneous solves Q=k-1', function () {
  var y = [1.2, -0.9, 1.5, -1.1];
  var v = [0.02, 0.02, 0.03, 0.02];
  var t = pauleMandelTau2(y, v);
  assert.ok(t > 0, 'tau2 positive for heterogeneous data');
  // independently verified against bisection: Q(tau2)=k-1 at tau2~=1.839365
  approx(t, 1.839365, 1e-4, 'tau2 matches bisection ground truth');
  // verify the moment condition Q(tau2) == k-1
  var w = v.map(function (vi) { return 1 / (vi + t); });
  var sw = w.reduce(function (a, b) { return a + b; }, 0);
  var yb = 0; for (var i = 0; i < y.length; i++) yb += y[i] * w[i]; yb /= sw;
  var Q = 0; for (var j = 0; j < y.length; j++) { var d = y[j] - yb; Q += w[j] * d * d; }
  approx(Q, y.length - 1, 1e-3, 'Q at solution equals k-1');
});

// 7. inverse-variance pooling: equal weights -> simple mean, narrower SE than any study
ok('poolWith inverse-variance', function () {
  var p = poolWith([0.1, 0.3], [0.04, 0.04], 0);
  approx(p.effect, 0.2, 1e-12, 'pooled mean');
  // se = 1/sqrt(1/0.04 + 1/0.04) = 1/sqrt(50)
  approx(p.se, 1 / Math.sqrt(50), 1e-12, 'pooled se');
  assert.ok(p.lo < p.effect && p.hi > p.effect, 'CI brackets estimate');
});

// 8. k=1 pooling does not crash and returns the single study unchanged
ok('poolWith k=1 returns single study', function () {
  var p = poolWith([0.5], [0.04], 0);
  approx(p.effect, 0.5, 1e-12, 'k=1 effect');
  approx(p.se, 0.2, 1e-12, 'k=1 se');
  assert.strictEqual(pauleMandelTau2([0.5], [0.04]), 0, 'k=1 tau2 is 0');
});

console.log('\n' + passed + ' passed');
