/* /play: a playable Sokoban over the classic level set.
 *
 * Crates are an array of positions plus a key -> index map, not a Set. The
 * index IS the crate's identity and never changes, so crate i always owns
 * sprite i. An earlier version kept crates in a Set and re-paired elements
 * to positions on every render; because deleting and re-adding a moved
 * crate sends it to the end of Set iteration order, the pairing shifted and
 * crates visibly swapped places with each other.
 *
 * Progress lives in localStorage, not cookies: it is only ever read by this
 * page, so there is no reason to attach it to every request to the server.
 */
(function () {
  "use strict";

  // Bumped to v3 when /play switched to the original Thinking Rabbit set:
  // progress is keyed by level index, so old saves would mark unrelated
  // levels solved.
  var STORE = "latent-sokoban.play.v3";
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
  var state = null;          // { player, crates:[[r,c]], crateAt:{}, ... }
  var level = null;
  var index = 0;
  var sprites = [];          // sprites[i] belongs to crates[i], always
  var playerEl = null;
  var progress = load();
  // Bumped by every loadLevel. A pending win callback carries the value it
  // was scheduled under and does nothing if the board has moved on since.
  var generation = 0;
  var winTimer = null;

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

  function indexCrates(crates) {
    var at = Object.create(null);
    for (var i = 0; i < crates.length; i++) at[key(crates[i][0], crates[i][1])] = i;
    return at;
  }

  function loadLevel(i) {
    generation++;                 // strands any win still waiting on a slide
    clearTimeout(winTimer);
    index = Math.max(0, Math.min(i, levels.length - 1));
    level = levels[index];
    progress.current = index;
    save();

    var crates = level.crates.map(function (p) { return [p[0], p[1]]; });
    state = {
      player: level.player.slice(),
      facing: "down",
      crates: crates,
      crateAt: indexCrates(crates),
      moves: 0, pushes: 0, history: []
    };
    level._walls = new Set(level.walls.concat(level.outer_walls)
      .map(function (p) { return key(p[0], p[1]); }));
    level._floor = new Set(level.floor.map(function (p) { return key(p[0], p[1]); }));
    level._goals = level.goals.map(function (p) { return key(p[0], p[1]); });

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

    var goals = new Set(level._goals);
    for (var r = 0; r < level.h; r++) {
      for (var c = 0; c < level.w; c++) {
        var k = key(r, c), d = document.createElement("div");
        if (level._walls.has(k)) {
          d.className = "cell wall";
        } else if (level._floor.has(k)) {
          d.className = goals.has(k) ? "cell goalcell" : "cell";
        } else {
          d.className = "cell void";
        }
        el.board.appendChild(d);
      }
    }

    sprites = state.crates.map(function () {
      var s = document.createElement("div");
      s.className = "sprite crate";
      el.board.appendChild(s);
      return s;
    });

    playerEl = document.createElement("div");
    playerEl.className = "sprite dot down";
    el.board.appendChild(playerEl);
  }

  /* Cell size is solved for, not guessed: take the smaller of the width and
     height each axis can afford, so the whole board is always on screen. */
  function fit() {
    var cs = getComputedStyle(el.stage);
    var w = el.stage.clientWidth
          - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    var h = el.stage.clientHeight
          - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
    var cell = Math.floor(Math.min(w / level.w, h / level.h));
    cell = Math.max(4, Math.min(cell, 64));
    el.board.style.setProperty("--cell", cell + "px");
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
    var goals = new Set(level._goals);
    for (var i = 0; i < state.crates.length; i++) {
      var r = state.crates[i][0], c = state.crates[i][1];
      sprites[i].classList.toggle("home", goals.has(key(r, c)));
      place(sprites[i], r, c, false);
    }
    place(playerEl, state.player[0], state.player[1], state.facing === "right");

    el.lvl.textContent = (index + 1) + " / " + levels.length;
    el.moves.textContent = state.moves;
    el.pushes.textContent = state.pushes;
    var b = progress.solved[index];
    el.best.textContent = b ? b.moves : "-";
    el.undo.disabled = state.history.length === 0;
  }

  // ------------------------------------------------------------------ moves

  /* True while the board must not change: no level yet, the win card is up,
     or a winning crate is still sliding. Undo has to respect this too, or a
     click during the slide rewinds the very push the win is waiting on. */
  function busy() {
    return !state || !el.win.hidden || state.locked;
  }

  function move(dir) {
    if (busy()) return;
    var d = DIRS[dir];
    if (!d) return;

    state.facing = dir;
    // The side sprite faces left, so it is the rightward move that gets
    // mirrored. Both directions share the one background image.
    playerEl.className = "sprite dot " + (dir === "right" ? "left" : dir);

    var tr = state.player[0] + d[0], tc = state.player[1] + d[1];
    var tk = key(tr, tc);

    // Walls and the void beyond the map both block.
    if (level._walls.has(tk) || !level._floor.has(tk)) { render(); return; }

    var snapshot = {
      player: state.player.slice(),
      crates: state.crates.map(function (p) { return [p[0], p[1]]; }),
      moves: state.moves, pushes: state.pushes
    };

    var pushed = -1;
    var ci = state.crateAt[tk];
    if (ci !== undefined) {
      var br = tr + d[0], bc = tc + d[1], bk = key(br, bc);
      if (level._walls.has(bk) || !level._floor.has(bk) ||
          state.crateAt[bk] !== undefined) {
        render(); return;                      // crate has nowhere to go
      }
      // Move crate ci in place: its index, and so its sprite, is unchanged.
      state.crates[ci] = [br, bc];
      delete state.crateAt[tk];
      state.crateAt[bk] = ci;
      state.pushes++;
      pushed = ci;
    }

    state.player = [tr, tc];
    state.moves++;
    state.history.push(snapshot);
    if (state.history.length > 5000) state.history.shift();

    render();
    if (solved()) winWhenTheCrateLands(pushed);
  }

  /* Announcing the solve on the same frame as the winning push covers the
     board before the crate has visibly arrived. Wait for that crate's slide
     to finish, then declare it. Input stays locked in between so the win
     cannot be moved out from under. */
  function winWhenTheCrateLands(ci) {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches || ci < 0) {
      win();
      return;
    }
    state.locked = true;
    var node = sprites[ci];
    var gen = generation;
    var fired = false;
    var finish = function () {
      if (fired) return;
      fired = true;
      node.removeEventListener("transitionend", finish);
      // Restarting or switching levels mid-slide leaves this scheduled
      // against a board that no longer exists; declaring a win then would
      // record the new level as solved in zero moves.
      if (gen !== generation) return;
      state.locked = false;
      win();
    };
    node.addEventListener("transitionend", finish);
    // transitionend does not fire if the transform did not actually change
    // or the tab is backgrounded, so never wait on it alone.
    winTimer = setTimeout(finish, 400);
  }

  function solved() {
    for (var i = 0; i < level._goals.length; i++) {
      if (state.crateAt[level._goals[i]] === undefined) return false;
    }
    return true;
  }

  function undo() {
    if (busy()) return;
    var s = state.history.pop();
    if (!s) return;
    state.player = s.player;
    state.crates = s.crates;
    state.crateAt = indexCrates(s.crates);
    state.moves = s.moves;
    state.pushes = s.pushes;
    render();
  }

  function win() {
    var prev = progress.solved[index];
    var record = !prev || state.moves < prev.moves;
    if (record) progress.solved[index] = { moves: state.moves, pushes: state.pushes };
    save();

    var plural = function (n, word) { return n + " " + word + (n === 1 ? "" : "s"); };
    el.winText.textContent =
      plural(state.moves, "move") + ", " + plural(state.pushes, "push") +
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
  function refit() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () { if (level) { fit(); render(); } }, 80);
  }
  addEventListener("resize", refit);
  /* The stage also changes height without the window resizing: the deck wraps
     to another row, a mobile URL bar slides away. The board is sized from the
     stage, never the other way round, so watching it cannot feed back. */
  if (window.ResizeObserver) new ResizeObserver(refit).observe(el.stage);

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
