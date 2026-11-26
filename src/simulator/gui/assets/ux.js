/* ux.js — camada de UX sobre o app_web.py. Carregar DEPOIS do script original. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const debounce = (f, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); }; };
  const sb = $('#sidebar');

  /* ---------- 1. Seed travada: só "Novo cenário" sorteia ---------- */
  const rollSeed = () => { $('#param-seed').value = Math.floor(Math.random() * 1e6); };
  const _getParams = window.getParams;
  window.getParams = () => {
    const s = $('#param-seed');
    if (!s.value || parseInt(s.value) < 0) rollSeed();
    return _getParams();
  };
  $('label[for=param-seed]').textContent = 'Seed (fixa):';

  /* ---------- 2. Debounce + persistência ---------- */
  const KEY = 'sim-ui-v1';
  const fields = () => $$('#sidebar input[id], #sidebar select[id]').filter(e => !e.id.startsWith('ux-'));
  const dump = () => Object.fromEntries(fields().map(e => [e.id, e.type === 'checkbox' ? e.checked : e.value]));
  const load = o => {
    for (const [k, v] of Object.entries(o)) {
      const e = document.getElementById(k); if (!e) continue;
      e.type === 'checkbox' ? (e.checked = !!v) : (e.value = v);
    }
    applyRules();
  };
  const persist = () => { try { localStorage.setItem(KEY, JSON.stringify(dump())); } catch (e) {} };

  const _upd = window.updateSnapshot;
  const run = () => { persist(); return _upd(); };
  window.updateSnapshot = debounce(run, 350);
  const newBtn = $('#sidebar .btn-accent');
  newBtn.onclick = () => { rollSeed(); run(); };

  /* clamp de números antes do handler inline */
  document.addEventListener('change', e => {
    const t = e.target;
    if (t.type !== 'number' || t.value === '') return;
    if (t.min !== '' && +t.value < +t.min) t.value = t.min;
    if (t.max !== '' && +t.value > +t.max) t.value = t.max;
  }, true);

  /* ---------- 3. Campos condicionais ---------- */
  const RULES = () => {
    const m = $('#param-model').value, p = $('#param-power').value, ad = m.startsWith('adaptive');
    return {
      'param-J': m === 'smb', 'param-L': m === 'smb' || ad, 'param-aoa-err': ad,
      'param-beamwidth': m === 'sector', 'param-g-min': m === 'sector',
      'param-sigma-r': $('#param-rayleigh').checked,
      'param-p-min': p !== 'full_power', 'param-gamma-tar': p !== 'full_power',
      'param-mu-step': p !== 'full_power', 'param-t-inner': p !== 'full_power',
      'param-tol': p === 'dual_loop', 'param-t-outer': p === 'dual_loop',
    };
  };
  function applyRules() {
    for (const [id, show] of Object.entries(RULES())) {
      const el = document.getElementById(id);
      if (el) el.parentElement.classList.toggle('ux-hidden', !show);
    }
  }
  sb.addEventListener('change', applyRules);

  /* ---------- 4. Ferramentas: busca, presets, salvar/carregar ---------- */
  const PRESETS = {
    'Ex.1 – Setorizado (M=16, K=4)': { 'param-M': 16, 'param-K': 4, 'param-N': 1, 'param-model': 'sector', 'param-power': 'full_power' },
    'Ex.2 – SMB (M=16, K=4)': { 'param-M': 16, 'param-K': 4, 'param-N': 1, 'param-model': 'smb', 'param-power': 'full_power' },
    'Ex.3 – MRC ideal (M=16, K=4)': { 'param-M': 16, 'param-K': 4, 'param-N': 1, 'param-model': 'adaptive', 'param-power': 'full_power' },
    'Ex.4 – MRC erro 5° (M=16, K=4)': { 'param-M': 16, 'param-K': 4, 'param-N': 1, 'param-model': 'adaptive_5', 'param-power': 'full_power' },
    'SciChal4A – Dual-Loop (M=9, K=4)': { 'param-M': 9, 'param-K': 4, 'param-N': 1, 'param-model': 'adaptive', 'param-power': 'dual_loop' },
  };
  const tools = document.createElement('div');
  tools.className = 'ux-tools';
  tools.innerHTML = `
    <input type="text" id="ux-search" placeholder="🔍 Buscar parâmetro…  (ex.: potência, L, seed)">
    <select id="ux-preset"><option value="">⚙️ Presets dos exercícios…</option>
      ${Object.keys(PRESETS).map(k => `<option>${k}</option>`).join('')}</select>
    <div class="row">
      <button class="btn mini" id="ux-save" title="Baixar configuração (JSON)">💾 Salvar</button>
      <button class="btn mini" id="ux-load" title="Carregar configuração (JSON)">📂 Abrir</button>
      <button class="btn mini" id="ux-reset" title="Restaurar padrões">↺ Padrão</button>
    </div>
    <input type="file" id="ux-file" accept=".json" hidden>`;
  sb.children[0].after(tools);

  $('#ux-preset').onchange = e => { const p = PRESETS[e.target.value]; if (p) { load(p); run(); } e.target.value = ''; };
  $('#ux-save').onclick = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(dump(), null, 2)], { type: 'application/json' }));
    a.download = 'config_simulador.json'; a.click();
  };
  $('#ux-load').onclick = () => $('#ux-file').click();
  $('#ux-file').onchange = async e => {
    try { load(JSON.parse(await e.target.files[0].text())); run(); toast('Configuração carregada.', 'ok'); }
    catch (err) { toast('Erro: JSON de configuração inválido.'); }
    e.target.value = '';
  };
  $('#ux-reset').onclick = () => { try { localStorage.removeItem(KEY); } catch (e) {} location.reload(); };

  $('#ux-search').oninput = e => {
    const q = e.target.value.trim().toLowerCase();
    $$('details.group-box').forEach(g => {
      let any = false;
      $$('.group-body label', g).forEach(l => {
        const cell = l.closest('.grid-2 > div, .grid-3 > div, .group-body > div') || l;
        const hit = !q || l.textContent.toLowerCase().includes(q) || g.querySelector('summary').textContent.toLowerCase().includes(q);
        cell.classList.toggle('ux-nomatch', !hit); any = any || hit;
      });
      g.classList.toggle('ux-nomatch', !any);
      if (q && any) g.open = true;
    });
  };

  /* ---------- 5. Barra de ações fixa, painel recolhível ---------- */
  sb.lastElementChild.classList.add('ux-actions');
  const tabsBar = $('#tabs-bar');
  const toggle = document.createElement('button');
  toggle.className = 'ux-ghost'; toggle.textContent = '⇤ Painel'; toggle.title = 'Ocultar/mostrar painel  ( [ )';
  const toggleSb = () => {
    document.body.classList.toggle('sb-collapsed');
    setTimeout(() => $$('iframe').forEach(f => { try { f.contentWindow.dispatchEvent(new Event('resize')); } catch (e) {} }), 250);
  };
  toggle.onclick = toggleSb; tabsBar.append(toggle);

  /* ---------- 6. Loading overlay + toasts ---------- */
  const overlay = document.createElement('div');
  overlay.id = 'ux-loading'; overlay.innerHTML = '<span>⏳ Processando…</span>';
  $('#tab-contents').append(overlay);
  const toasts = document.createElement('div'); toasts.id = 'ux-toasts'; document.body.append(toasts);
  function toast(msg, type = 'err') {
    const t = document.createElement('div');
    t.className = 'ux-toast ' + type; t.textContent = msg; toasts.append(t);
    setTimeout(() => t.remove(), type === 'err' ? 7000 : 3000);
  }
  const _setStatus = window.setStatus;
  window.setStatus = (text, loading = false) => {
    _setStatus(text, loading);
    overlay.classList.toggle('on', loading);
    if (/^Erro/.test(text)) toast(text);
  };

  /* ---------- 7. Tabela: ordenar, filtrar outage, selecionar ---------- */
  const sortSt = { i: -1, dir: 1 };
  const num = t => { const m = t.match(/-?\d+(\.\d+)?/); return m ? parseFloat(m[0]) : NaN; };
  function applySort() {
    if (sortSt.i < 0) return;
    const tb = $('#users-tbody');
    [...tb.rows].sort((a, b) => {
      const ta = a.cells[sortSt.i].textContent, tb2 = b.cells[sortSt.i].textContent, x = num(ta), y = num(tb2);
      if (isNaN(x) && isNaN(y)) return ta.localeCompare(tb2) * sortSt.dir;
      return ((isNaN(x) ? -Infinity : x) - (isNaN(y) ? -Infinity : y)) * sortSt.dir;
    }).forEach(r => tb.appendChild(r));
  }
  const applyFilter = () => {
    const only = $('#ux-outage')?.checked;
    $$('#users-tbody tr').forEach(r => { r.hidden = !!only && !/Outage/.test(r.cells[11].textContent); });
  };
  $$('#users-table th').forEach((th, i) => {
    th.classList.add('sortable');
    th.onclick = () => {
      sortSt.dir = sortSt.i === i ? -sortSt.dir : 1; sortSt.i = i;
      $$('#users-table th').forEach(h => h.classList.remove('asc', 'desc'));
      th.classList.add(sortSt.dir > 0 ? 'asc' : 'desc'); applySort();
    };
  });
  const bar = document.createElement('div');
  bar.className = 'ux-bar';
  bar.innerHTML = '<label class="checkbox-row"><input type="checkbox" id="ux-outage"><span>Somente outage</span></label><span>Clique no cabeçalho para ordenar</span>';
  $('#users-summary').parentElement.append(bar);
  $('#ux-outage').onchange = applyFilter;
  $('#users-tbody').addEventListener('click', e => {
    const tr = e.target.closest('tr'); if (!tr) return;
    $$('#users-tbody tr.sel').forEach(r => r.classList.remove('sel')); tr.classList.add('sel');
  });
  const _render = window.renderUsersTable;
  window.renderUsersTable = u => { _render(u); applySort(); applyFilter(); };

  /* ---------- 8. Abas acessíveis + atalhos ---------- */
  tabsBar.setAttribute('role', 'tablist');
  const tabs = $$('.tab-btn');
  const markTabs = idx => tabs.forEach((b, i) => { b.setAttribute('role', 'tab'); b.setAttribute('aria-selected', i === idx); b.tabIndex = i === idx ? 0 : -1; });
  markTabs(0);
  const _switch = window.switchTab;
  window.switchTab = i => { _switch(i); markTabs(i); };
  tabsBar.addEventListener('keydown', e => {
    const cur = tabs.findIndex(b => b.getAttribute('aria-selected') === 'true');
    if (e.key === 'ArrowRight') { switchTab((cur + 1) % tabs.length); tabs[(cur + 1) % tabs.length].focus(); }
    if (e.key === 'ArrowLeft') { const n = (cur + tabs.length - 1) % tabs.length; switchTab(n); tabs[n].focus(); }
  });
  addEventListener('keydown', e => {
    if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName) || e.ctrlKey || e.metaKey || e.altKey) return;
    if (/^[1-4]$/.test(e.key)) switchTab(+e.key - 1);
    else if (e.key === 'r' || e.key === 'R') newBtn.click();
    else if (e.key === '[') toggleSb();
    else if (e.key === '?') toast('Atalhos: 1–4 abas · R novo cenário · [ ocultar painel', 'ok');
  });

  /* ---------- init ---------- */
  try { load(JSON.parse(localStorage.getItem(KEY) || '{}')); } catch (e) {}
  applyRules();
})();
