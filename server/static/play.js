/* /play: a playable Sokoban over the classic level set.
 *
 * Board state is three sets of "r,c" keys plus the player position. Moves
 * push an undo frame first, so undo is a pop rather than a replay.
 *
 * Progress lives in localStorage, not cookies: it is only ever read by this
 * page, so there is no reason to attach it to every request to the server.
 */
(function () {
  "use strict";

  var STORE = "latent-sokoban.play.v1";
  var DIRS = {
    up:    [-1, 0], down:  [1, 0],
    left:  [0, -1], right: [0, 1]
  };

  var el = {
    stage:   document.getElementById("stage"),
    board:   document.getElementById("board"),
    lvl:     document.getElementById("lvl"),
    moves:   document.getElementById("moves"),
    pushes:  document.getElementById("pushes"),
    best:    document.getElementById("best"),
    undo:    document.getElementById("undo"),
    restart: document.getElementById("restart"),
    picker:  document.getElementById("picker"),
    pickerGrid: document.getElementById("picker-grid"),
    pickBtn: document.getElementById("pick"),
    win:     document.getElementById("win"),
    winText: document.getElementById("win-text"),
    next:    document.getElementById("next"),
    status:  document.getElementById("status")
  };

  var levels = [];
  var state = null;          // { player, crates:Set, moves, pushes, history:[] }
  var level = null;          // the current level record
  var index = 0;
  var sprites = {};          // "r,c" -> crate element
  var playerEl = null;
  var progress = load();

  // ---------------------------------------------------------------- storage

  function load() {
    try {
      var raw = localStorage.getItem(STORE);
      if (!raw) return { solved: {}, current: 0 };
      var p = JSON.parse(raw);
      return { solved: p.solved || {}, current: p.current || 0 };
    } catch (e) {
      // Private mode, disabled storage, or corrupt JSON: play without
      // persistence rather than failing to start.
      return { solved: {}, current: 0 };
    }
  }

  function save() {
    try {
      localStorage.setItem(STORE, JSON.stringify(progress));
    } catch (e) { /* storage full or blocked; the game still plays */ }
  }

  // ------------------------------------------------------------------ setup

  function key(r, c) { return r + "," + c; }

  function loadLevel(i) {
    index = Math.max(0, Math.min(i, levels.length - 1));
    level = levels[index];
    progress.current = index;
    save();

    state = {
      player: level.player.slice(),
      crates: new Set(level.crates.map(function (p) { return key(p[0], p[1]); })),
      moves: 0, pushes: 0, history: []
    };
    level._walls = new Set(level.walls.concat(level.outer_walls)
      .map(function (p) { return key(p[0], p[1]); }));
    level._floor = new Set(level.floor.map(function (p) { return key(p[0], p[1]); }));
    level._goals = new Set(level.goals.map(function (p) { return key(p[0], p[1]); }));

    build();
    fit();
    render();
    el.win.hidden = true;
    announce("Level " + (index + 1) + " of " + levels.length + ", " +
             level.crates.length + " crates");
  }

  function build() {
    el.board.textContent = "";
    el.board.style.setProperty("--cols", level.w);

    var outer = new Set(level.outer_walls.map(function (p) { return key(p[0], p[1]); }));
    for (var r = 0; r < level.h; r++) {
      for (var c = 0; c < level.w; c++) {
        var k = key(r, c), d = document.createElement("div");
        if (level._walls.has(k)) {
          d.className = outer.has(k) ? "cell wall outer" : "cell wall";
        } else if (level._floor.has(k)) {
          d.className = level._goals.has(k) ? "cell goalcell" : "cell";
        } else {
          d.className = "cell void";
        }
        el.board.appendChild(d);
      }
    }

    sprites = {};
    state.crates.forEach(function (k) {
      var s = document.createElement("div");
      s.className = "sprite crate";
      el.board.appendChild(s);
      sprites[k] = s;
    });

    playerEl = document.createElement("div");
    playerEl.className = "sprite dot down";
    el.board.appendChild(playerEl);
  }

  /* Cell size is solved for, not guessed: take the smaller of the width and
     height each axis can afford, so the whole board is always on screen. */
  function fit() {
    var box = el.stage.getBoundingClientRect();
    var pad = 24;
    var cell = Math.floor(Math.min(
      (box.width  - pad) / level.w,
      (box.height - pad) / level.h
    ));
    cell = Math.max(9, Math.min(cell, 64));
    el.board.style.setProperty("--cell", cell + "px");
    return cell;
  }

  function cellSize() {
    return parseFloat(getComputedStyle(el.board).getPropertyValue("--cell")) || 32;
  }

  // ----------------------------------------------------------------- render

  function place(node, r, c, flip) {
    var s = cellSize();
    node.style.transform = "translate(" + (c * s) + "px," + (r * s) + "px)" +
                           (flip ? " scaleX(-1)" : "");
  }

  function render() {
    var moved = Object.create(null);
    state.crates.forEach(function (k) { moved[k] = true; });

    // Re-key the crate elements to their current squares.
    var pool = [];
    for (var old in sprites) pool.push(sprites[old]);
    var i = 0, fresh = {};
    state.crates.forEach(function (k) {
      var parts = k.split(","), r = +parts[0], c = +parts[1];
      var node = pool[i++];
      node.classList.toggle("home", level._goals.has(k));
      place(node, r, c, false);
      fresh[k] = node;
    });
    sprites = fresh;

    place(playerEl, state.player[0], state.player[1], state.facing === "left");

    el.lvl.textContent = (index + 1) + " / " + levels.length;
    el.moves.textContent = state.moves;
    el.pushes.textContent = state.pushes;
    var b = progress.solved[index];
    el.best.textContent = b ? b.moves : "-";
    el.undo.disabled = state.history.length === 0;
  }

  // ------------------------------------------------------------------ moves

  function move(dir) {
    if (!state || !el.win.hidden) return;
    var d = DIRS[dir];
    if (!d) return;

    state.facing = dir;
    playerEl.className = "sprite dot " + (dir === "left" ? "right" : dir);

    var pr = state.player[0], pc = state.player[1];
    var tr = pr + d[0], tc = pc + d[1], tk = key(tr, tc);

    // Walls and the void beyond the map both block.
    if (level._walls.has(tk) || !level._floor.has(tk)) { render(); return; }

    var snapshot = {
      player: state.player.slice(),
      crates: new Set(state.crates),
      moves: state.moves, pushes: state.pushes
    };

    if (state.crates.has(tk)) {
      var br = tr + d[0], bc = tc + d[1], bk = key(br, bc);
      if (level._walls.has(bk) || !level._floor.has(bk) || state.crates.has(bk)) {
        render(); return;                      // crate has nowhere to go
      }
      state.crates.delete(tk);
      state.crates.add(bk);
      state.pushes++;
    }

    state.player = [tr, tc];
    state.moves++;
    state.history.push(snapshot);
    if (state.history.length > 2000) state.history.shift();

    render();
    if (solved()) win();
  }

  function solved() {
    var all = true;
    level._goals.forEach(function (g) { if (!state.crates.has(g)) all = false; });
    return all;
  }

  function undo() {
    var s = state.history.pop();
    if (!s) return;
    state.player = s.player; state.crates = s.crates;
    state.moves = s.moves; state.pushes = s.pushes;
    render();
  }

  function win() {
    var prev = progress.solved[index];
    var record = !prev || state.moves < prev.moves;
    if (record) progress.solved[index] = { moves: state.moves, pushes: state.pushes };
    save();

    el.winText.textContent = state.moves + " moves, " + state.pushes + " pushes" +
      (record && prev ? ", a new best, beating " + prev.moves : "");
    el.win.hidden = false;
    el.next.disabled = index >= levels.length - 1;
    el.next.focus();
    announce("Solved in " + state.moves + " moves");
    drawPicker();
  }

  function announce(msg) { el.status.textContent = msg; }

  // ---------------------------------------------------------------- picker

  function drawPicker() {
    el.pickerGrid.textContent = "";
    for (var i = 0; i < levels.length; i++) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = String(i + 1);
      if (progress.solved[i]) {
        b.classList.add("solved");
        b.title = "Solved in " + progress.solved[i].moves + " moves";
      }
      if (i === index) b.classList.add("current");
      b.setAttribute("aria-label", "Level " + (i + 1) +
        (progress.solved[i] ? ", solved in " + progress.solved[i].moves + " moves" : ""));
      (function (n) {
        b.addEventListener("click", function () {
          el.picker.hidden = true;
          loadLevel(n);
        });
      })(i);
      el.pickerGrid.appendChild(b);
    }
  }

  // ------------------------------------------------------------------ input

  var KEYS = {
    ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
    w: "up", s: "down", a: "left", d: "right",
    W: "up", S: "down", A: "left", D: "right"
  };

  addEventListener("keydown", function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (!el.picker.hidden && e.key === "Escape") { el.picker.hidden = true; return; }
    var dir = KEYS[e.key];
    if (dir) { e.preventDefault(); move(dir); return; }
    if (e.key === "u" || e.key === "U" || e.key === "z" || e.key === "Z") {
      e.preventDefault(); undo();
    } else if (e.key === "r" || e.key === "R") {
      e.preventDefault(); loadLevel(index);
    } else if (e.key === "n" || e.key === "N") {
      e.preventDefault(); if (index < levels.length - 1) loadLevel(index + 1);
    } else if (e.key === "p" || e.key === "P") {
      e.preventDefault(); if (index > 0) loadLevel(index - 1);
    }
  });

  /* Swipe: a short flick in a clear direction is one move. The dominant
     axis wins so a sloppy diagonal still does what was meant. */
  var touch = null;
  el.stage.addEventListener("pointerdown", function (e) {
    touch = { x: e.clientX, y: e.clientY };
  });
  el.stage.addEventListener("pointerup", function (e) {
    if (!touch) return;
    var dx = e.clientX - touch.x, dy = e.clientY - touch.y;
    touch = null;
    if (Math.abs(dx) < 24 && Math.abs(dy) < 24) return;
    if (Math.abs(dx) > Math.abs(dy)) move(dx > 0 ? "right" : "left");
    else move(dy > 0 ? "down" : "up");
  });
  el.stage.addEventListener("pointercancel", function () { touch = null; });

  Array.prototype.forEach.call(document.querySelectorAll("[data-dir]"), function (b) {
    b.addEventListener("click", function () { move(b.getAttribute("data-dir")); });
  });

  el.undo.addEventListener("click", undo);
  el.restart.addEventListener("click", function () { loadLevel(index); });
  el.next.addEventListener("click", function () {
    if (index < levels.length - 1) loadLevel(index + 1);
  });
  el.pickBtn.addEventListener("click", function () {
    drawPicker();
    el.picker.hidden = !el.picker.hidden;
  });
  document.getElementById("picker-close")
    .addEventListener("click", function () { el.picker.hidden = true; });

  var resizeTimer;
  addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () { if (level) { fit(); render(); } }, 80);
  });

  // ------------------------------------------------------------------- boot

  fetch("/static/play-levels.json")
    .then(function (r) {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    })
    .then(function (data) {
      levels = data.levels;
      drawPicker();
      loadLevel(progress.current || 0);
    })
    .catch(function () {
      el.stage.textContent = "Could not load the levels. Reload to try again.";
    });
})();
