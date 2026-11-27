import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { StrictMode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProblemEntry, ProblemList, ProjectTarget } from './api/client'
import { ThemeProvider } from './lib/theme'
import { useTheme } from './lib/themeContext'
import { StageApp, type StageAppProps } from './StageApp'

/** 問題の設定の置き場 */
const ROOT = '/home/user/.ml-pathfinder/problems'

/** 置き場の問題の設定の 1 件 (reason があれば使えない) */
function entryOf(id: string, name: string, reason: string | null = null): ProblemEntry {
  const project = `/work/${id}`
  return {
    path: `${ROOT}/${id}/problem.yaml`,
    name,
    created_at: '2026-01-01T00:00:00',
    updated_at: '2026-01-02T00:00:00',
    usable: reason === null,
    reason,
    project: reason === null ? project : null,
    dataset: reason === null ? `${project}/outputs/dataset` : null,
    target: reason === null ? `${project}/models/model.yaml` : null,
    example: null,
  }
}

const PROBLEMS: ProblemList = {
  root: ROOT,
  problems: [
    entryOf('aaaaaaaaaaaa', '総菜の作る量'),
    entryOf('bbbbbbbbbbbb', '解約の予測'),
    entryOf('cccccccccccc', '書き出していない課題', 'プロジェクトを書き出していません'),
  ],
  error: null,
}

/** 問題の設定の受け持ちに合う雛形の定義 (入力から出力へ 1 つの部品でつなぐ) */
const TEMPLATE = {
  format_version: 1,
  name: 'tabular_mlp',
  inputs: { x: { shape: [8, 3], ui: { x: 40, y: 160 } } },
  nodes: {
    head: { unit: 'nn.Linear', args: { in_features: 3, out_features: 2 }, inputs: 'x' },
  },
  outputs: { logits: 'head' },
}

/** 確かめの結果 (空の定義) */
const CHECKED = {
  issues: [],
  inferred: true,
  ok: true,
  nodes: {},
  outputs: {},
  params: 0,
  trainable: 0,
  estimated_only: false,
  yaml: '',
}

/** 保存先を確かめた結果 (置き場の問題の設定か, 置き場の外のプロジェクトのディレクトリ) */
function targetOf(text: string): ProjectTarget {
  const entry = PROBLEMS.problems.find((item) => item.path === text)
  if (entry) {
    return {
      path: entry.target!,
      source: 'problem',
      exists: false,
      project: entry.project,
      dataset: entry.dataset,
      problem: entry.name,
      problem_path: entry.path,
      example: null,
      default: false,
    }
  }
  return {
    path: `${text}/models/model.yaml`,
    source: 'problem',
    exists: false,
    project: text,
    dataset: `${text}/outputs/dataset`,
    problem: '写し',
    problem_path: `${text}/problem.yaml`,
    example: null,
    default: false,
  }
}

/** API の頭 `/api/model` の下だけに答える fetch (他の場所を呼べば 404) */
function stubApi(base = '/api/model', example: string | null = null) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    const method = init?.method ?? 'GET'
    if (!url.startsWith(`${base}/`)) return new Response(null, { status: 404 })
    const path = url.slice(base.length)
    if (path === '/units') return Response.json([])
    if (path === '/check') return Response.json(CHECKED)
    if (path === '/models' || path === '/examples') return Response.json([])
    if (path === '/problems') return Response.json(PROBLEMS)
    if (path === '/project/target' && method === 'POST') {
      const { path: text } = JSON.parse(String(init?.body)) as { path: string }
      return Response.json({ ...targetOf(text), example })
    }
    if (example !== null && path === `/examples/${example}`) {
      return Response.json({ name: example, spec: TEMPLATE })
    }
    if (path === '/project/model' && method === 'PUT') {
      const { target } = JSON.parse(String(init?.body)) as { target: string }
      return Response.json({ path: target, updated_at: '2026-01-03T00:00:00', notes: [] })
    }
    return new Response(null, { status: 404 })
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

/** ブラウザに残った下書き (1 つだけあることも確かめる) */
function onlyDraft(): { spec: { inputs: object; nodes: object } } {
  const drafts = Object.values(
    JSON.parse(window.localStorage.getItem('ml-createarchitecture:drafts') ?? '{}') as Record<
      string,
      { spec: { inputs: object; nodes: object } }
    >,
  )
  expect(drafts).toHaveLength(1)
  return drafts[0]!
}

/** 殻の下の形 (経路の頭 `/model`, API の頭 `/api/model`) で描く */
function renderStage(props: Partial<StageAppProps> = {}) {
  const onProjectChange = vi.fn()
  const onProjectsChange = vi.fn()
  const view = render(
    <StageApp
      basename="/model"
      apiBase="/api/model"
      project={null}
      onProjectChange={onProjectChange}
      onProjectsChange={onProjectsChange}
      embedded
      {...props}
    />,
  )
  return { ...view, onProjectChange, onProjectsChange }
}

/** 「データセットとプロジェクト」のダイアログを開く */
async function openProjectDialog() {
  fireEvent.click(await screen.findByRole('button', { name: 'データセットとプロジェクト' }))
  return screen.findByRole('dialog', { name: 'データセットとプロジェクト', hidden: true })
}

beforeEach(() => {
  // jsdom は対話の枠を開けず, 大きさの変化も見られないので, 開いた印と空の見張りを置く
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true
  }
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  )
  window.localStorage.clear()
  document.documentElement.classList.remove('dark')
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('StageApp', () => {
  it('API を全て apiBase の下に呼ぶ', async () => {
    const fetchMock = stubApi()
    renderStage()
    await screen.findByRole('button', { name: 'データセットとプロジェクト' })
    await waitFor(() =>
      expect(fetchMock.mock.calls.some(([input]) => String(input) === '/api/model/check')).toBe(
        true,
      ),
    )
    const urls = fetchMock.mock.calls.map(([input]) => String(input))
    for (const url of urls) expect(url.startsWith('/api/model/')).toBe(true)
  })

  it('embedded ではアプリの名前と明暗の切り替えを出さず, html のクラスを書き換えない', async () => {
    stubApi()
    document.documentElement.classList.add('dark')
    renderStage()
    await screen.findByRole('button', { name: 'データセットとプロジェクト' })
    expect(screen.queryByRole('heading', { name: 'ML-CreateArchitecture' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '暗い表示' })).not.toBeInTheDocument()
    // 殻が html のクラスを替えると, キャンバスの明暗 (React Flow の colorMode) も替わり,
    // 部品は html のクラスを書き戻さない
    const canvas = document.querySelector('.react-flow')!
    expect(canvas).toHaveClass('dark')
    document.documentElement.classList.remove('dark')
    await waitFor(() => expect(canvas).not.toHaveClass('dark'))
    expect(document.documentElement.classList.contains('dark')).toBe(false)
    document.documentElement.classList.add('dark')
    await waitFor(() => expect(canvas).toHaveClass('dark'))
  })

  it('読むだけの明暗は html のクラスを書き換えず, 殻の切り替えを読み直す', async () => {
    function Shown() {
      return <p>{useTheme().theme}</p>
    }
    render(
      <ThemeProvider readOnly>
        <Shown />
      </ThemeProvider>,
    )
    expect(screen.getByText('light')).toBeInTheDocument()
    expect(document.documentElement.classList.contains('dark')).toBe(false)
    document.documentElement.classList.add('dark')
    expect(await screen.findByText('dark')).toBeInTheDocument()
    document.documentElement.classList.remove('dark')
    expect(await screen.findByText('light')).toBeInTheDocument()
  })

  it('project を渡すと, その問題の設定の保存先とデータセットを既定にし, 同じ値は知らせない', async () => {
    const fetchMock = stubApi()
    const { onProjectChange } = renderStage({ project: 'aaaaaaaaaaaa' })
    const dialog = await openProjectDialog()
    const note = await within(dialog).findByRole('region', { name: '選んだ問題の設定' })
    expect(within(note).getByText(/総菜の作る量/)).toBeInTheDocument()
    await waitFor(() =>
      expect(
        within(dialog).getByRole('textbox', { name: 'データセットのディレクトリ' }),
      ).toHaveValue('/work/aaaaaaaaaaaa/outputs/dataset'),
    )
    expect(within(dialog).getByRole('textbox', { name: 'プロジェクトか問題の設定' })).toHaveValue(
      `${ROOT}/aaaaaaaaaaaa/problem.yaml`,
    )
    expect(within(dialog).getByText('/work/aaaaaaaaaaaa/models/model.yaml')).toBeInTheDocument()
    const targets = fetchMock.mock.calls.filter(
      ([input]) => String(input) === '/api/model/project/target',
    )
    expect(targets).toHaveLength(1)
    expect(onProjectChange).not.toHaveBeenCalled()
  })

  it('使えない問題の設定は理由を出し, 置き場に無い識別子は無いと出す', async () => {
    stubApi()
    const { rerender } = renderStage({ project: 'cccccccccccc' })
    const dialog = await openProjectDialog()
    expect(
      await within(dialog).findByText(/プロジェクトを書き出していません\. 下の一覧か欄/),
    ).toBeInTheDocument()
    expect(within(dialog).getByRole('textbox', { name: 'プロジェクトか問題の設定' })).toHaveValue(
      '',
    )
    rerender(
      <StageApp
        basename="/model"
        apiBase="/api/model"
        project="dddddddddddd"
        onProjectChange={() => {}}
        onProjectsChange={() => {}}
        embedded
      />,
    )
    expect(
      await within(dialog).findByText(/問題の設定 dddddddddddd は置き場 .* にありません/),
    ).toBeInTheDocument()
  })

  it('一覧から別の問題の設定を選ぶと, その識別子を 1 度だけ知らせる', async () => {
    stubApi()
    const { onProjectChange } = renderStage({ project: 'aaaaaaaaaaaa' })
    const dialog = await openProjectDialog()
    await waitFor(() =>
      expect(within(dialog).getByRole('textbox', { name: 'プロジェクトか問題の設定' })).toHaveValue(
        `${ROOT}/aaaaaaaaaaaa/problem.yaml`,
      ),
    )
    fireEvent.click(within(dialog).getByRole('button', { name: /解約の予測/ }))
    await waitFor(() => expect(onProjectChange).toHaveBeenCalledWith('bbbbbbbbbbbb'))
    expect(within(dialog).getByRole('textbox', { name: 'データセットのディレクトリ' })).toHaveValue(
      '/work/bbbbbbbbbbbb/outputs/dataset',
    )
    expect(onProjectChange).toHaveBeenCalledTimes(1)
  })

  it('StrictMode でも, 別の問題の設定を選んだ知らせは 1 度だけ', async () => {
    stubApi()
    const onProjectChange = vi.fn()
    render(
      <StrictMode>
        <StageApp
          basename="/model"
          apiBase="/api/model"
          project="aaaaaaaaaaaa"
          onProjectChange={onProjectChange}
          onProjectsChange={() => {}}
          embedded
        />
      </StrictMode>,
    )
    const dialog = await openProjectDialog()
    fireEvent.click(await within(dialog).findByRole('button', { name: /解約の予測/ }))
    await waitFor(() => expect(onProjectChange).toHaveBeenCalledWith('bbbbbbbbbbbb'))
    expect(onProjectChange).toHaveBeenCalledTimes(1)
  })

  it('殻が知らせた値を返しても入れ直さず, 置き場の外の場所では知らせない', async () => {
    const fetchMock = stubApi()
    const onProjectChange = vi.fn()
    const props = {
      basename: '/model',
      apiBase: '/api/model',
      onProjectChange,
      onProjectsChange: () => {},
      embedded: true,
    }
    const { rerender } = render(<StageApp {...props} project="aaaaaaaaaaaa" />)
    const dialog = await openProjectDialog()
    const field = within(dialog).getByRole('textbox', { name: 'プロジェクトか問題の設定' })
    await waitFor(() => expect(field).toHaveValue(`${ROOT}/aaaaaaaaaaaa/problem.yaml`))
    fireEvent.click(within(dialog).getByRole('button', { name: /解約の予測/ }))
    await waitFor(() => expect(onProjectChange).toHaveBeenCalledWith('bbbbbbbbbbbb'))
    // 殻が受け取った値を返す (同じ保存先を確かめ直さない)
    rerender(<StageApp {...props} project="bbbbbbbbbbbb" />)
    await waitFor(() => expect(field).toHaveValue(`${ROOT}/bbbbbbbbbbbb/problem.yaml`))
    const count = () =>
      fetchMock.mock.calls.filter(([input]) => String(input) === '/api/model/project/target').length
    expect(count()).toBe(2)

    // 置き場の外のプロジェクトを書いて確かめても知らせない
    fireEvent.change(field, { target: { value: '/backup/bbbbbbbbbbbb' } })
    fireEvent.click(within(dialog).getByRole('button', { name: '保存先を確かめる' }))
    expect(
      await within(dialog).findByText('/backup/bbbbbbbbbbbb/models/model.yaml'),
    ).toBeInTheDocument()
    expect(count()).toBe(3)
    expect(onProjectChange).toHaveBeenCalledTimes(1)
  })

  it('一覧を読み込み済みで開き直しても, 殻の問題の設定を既定にする', async () => {
    stubApi()
    renderStage({ project: 'aaaaaaaaaaaa' })
    let dialog = await openProjectDialog()
    const dataset = () =>
      within(dialog).getByRole('textbox', { name: 'データセットのディレクトリ' })
    await waitFor(() => expect(dataset()).toHaveValue('/work/aaaaaaaaaaaa/outputs/dataset'))
    fireEvent.change(dataset(), { target: { value: '/other' } })
    fireEvent.click(within(dialog).getByRole('button', { name: '閉じる' }))
    await waitFor(() => expect(dialog).not.toBeInTheDocument())
    // 覚えた場所からでなく, 殻の問題の設定から入ることを見るため, 覚えた場所を替える
    window.localStorage.setItem('ml-createarchitecture:project-path', '/remembered')
    window.localStorage.setItem('ml-createarchitecture:dataset-path', '/remembered')
    dialog = await openProjectDialog()
    await waitFor(() => expect(dataset()).toHaveValue('/work/aaaaaaaaaaaa/outputs/dataset'))
    expect(within(dialog).getByRole('textbox', { name: 'プロジェクトか問題の設定' })).toHaveValue(
      `${ROOT}/aaaaaaaaaaaa/problem.yaml`,
    )
  })

  it('知らせた値を殻が受け付けず, 後で殻がその値に選び直したら入れる', async () => {
    stubApi()
    const onProjectChange = vi.fn()
    const props = {
      basename: '/model',
      apiBase: '/api/model',
      onProjectChange,
      onProjectsChange: () => {},
      embedded: true,
    }
    const { rerender } = render(<StageApp {...props} project="aaaaaaaaaaaa" />)
    const dialog = await openProjectDialog()
    const field = within(dialog).getByRole('textbox', { name: 'プロジェクトか問題の設定' })
    await waitFor(() => expect(field).toHaveValue(`${ROOT}/aaaaaaaaaaaa/problem.yaml`))
    // B を選んで知らせ (殻は受け付けない), A に戻す (殻の選びと同じなので知らせない)
    fireEvent.click(within(dialog).getByRole('button', { name: /解約の予測/ }))
    await waitFor(() => expect(field).toHaveValue(`${ROOT}/bbbbbbbbbbbb/problem.yaml`))
    fireEvent.click(within(dialog).getByRole('button', { name: /総菜の作る量/ }))
    await waitFor(() => expect(field).toHaveValue(`${ROOT}/aaaaaaaaaaaa/problem.yaml`))
    // 同じ保存先を確かめ直しても知らせない
    fireEvent.click(within(dialog).getByRole('button', { name: '保存先を確かめる' }))
    await within(dialog).findByText('/work/aaaaaaaaaaaa/models/model.yaml')
    expect(onProjectChange.mock.calls).toEqual([['bbbbbbbbbbbb']])
    // 殻が B に選び直すと入れる
    rerender(<StageApp {...props} project="bbbbbbbbbbbb" />)
    await waitFor(() => expect(field).toHaveValue(`${ROOT}/bbbbbbbbbbbb/problem.yaml`))
    expect(within(dialog).getByRole('textbox', { name: 'データセットのディレクトリ' })).toHaveValue(
      '/work/bbbbbbbbbbbb/outputs/dataset',
    )
    expect(onProjectChange).toHaveBeenCalledTimes(1)
  })

  it('実験の定義の写しの場所の欄を, 殻の問題の設定のプロジェクトにする', async () => {
    stubApi()
    renderStage({ project: 'bbbbbbbbbbbb' })
    fireEvent.click(await screen.findByRole('button', { name: '開く' }))
    const dialog = await screen.findByRole('dialog', { name: '定義を開く', hidden: true })
    await waitFor(() =>
      expect(within(dialog).getByRole('textbox', { name: 'プロジェクトか問題の設定' })).toHaveValue(
        '/work/bbbbbbbbbbbb',
      ),
    )
  })

  it('プロジェクトに保存してから外すと下書きを残さず, 戻すと下書きを聞かない', async () => {
    stubApi('/api/model', 'tabular_mlp')
    const first = renderStage({ project: 'aaaaaaaaaaaa' })
    const dialog = await openProjectDialog()
    // 新しく始める定義なので, 殻の問題の設定の雛形を開く (保存していない変更になる)
    await within(dialog).findByText(/雛形 tabular_mlp を開きました/)
    const saving = within(dialog).getByRole('region', { name: 'プロジェクトに保存' })
    fireEvent.click(within(saving).getByRole('button', { name: 'この場所に保存' }))
    await within(saving).findByText('/work/aaaaaaaaaaaa/models/model.yaml に保存しました')
    // 置き場には保存していないので印 (*) は残る
    expect(screen.getByRole('button', { name: '保存 *' })).toBeInTheDocument()
    // 段階を移る (部品が外される) と, 書いたものと同じ定義は下書きに書かない
    first.unmount()
    const drafts = JSON.parse(
      window.localStorage.getItem('ml-createarchitecture:drafts') ?? '{}',
    ) as object
    expect(Object.keys(drafts)).toHaveLength(0)
    // 段階に戻っても下書きを戻すかを聞かない
    renderStage({ project: 'aaaaaaaaaaaa' })
    await screen.findByRole('button', { name: 'データセットとプロジェクト' })
    expect(
      screen.queryByRole('region', { name: '保存していない定義の下書き' }),
    ).not.toBeInTheDocument()
  })

  it('プロジェクトに保存した後に変えた定義は, 外すときに下書きに書く', async () => {
    stubApi('/api/model', 'tabular_mlp')
    const first = renderStage({ project: 'aaaaaaaaaaaa' })
    const dialog = await openProjectDialog()
    await within(dialog).findByText(/雛形 tabular_mlp を開きました/)
    const saving = within(dialog).getByRole('region', { name: 'プロジェクトに保存' })
    fireEvent.click(within(saving).getByRole('button', { name: 'この場所に保存' }))
    await within(saving).findByText('/work/aaaaaaaaaaaa/models/model.yaml に保存しました')
    fireEvent.click(within(dialog).getByRole('button', { name: '閉じる' }))
    // 保存した後に定義を変える (入力を加える)
    fireEvent.click(screen.getByRole('button', { name: '入力' }))
    first.unmount()
    // 下書きは変えた後の定義 (加えた入力を持つ)
    expect(Object.keys(onlyDraft().spec.inputs).sort()).toEqual(['input1', 'x'])
    renderStage({ project: 'aaaaaaaaaaaa' })
    expect(
      await screen.findByRole('region', { name: '保存していない定義の下書き' }),
    ).toBeInTheDocument()
  })

  it('プロジェクトに保存してから殻で問題の設定を替えると, 書いた定義も下書きに書く', async () => {
    stubApi('/api/model', 'tabular_mlp')
    const first = renderStage({ project: 'aaaaaaaaaaaa' })
    const dialog = await openProjectDialog()
    await within(dialog).findByText(/雛形 tabular_mlp を開きました/)
    const saving = within(dialog).getByRole('region', { name: 'プロジェクトに保存' })
    fireEvent.click(within(saving).getByRole('button', { name: 'この場所に保存' }))
    await within(saving).findByText('/work/aaaaaaaaaaaa/models/model.yaml に保存しました')
    fireEvent.click(within(dialog).getByRole('button', { name: '閉じる' }))
    // 替えた先のプロジェクトには書いていないので, 替えたときに待たずに下書きに書く
    first.rerender(
      <StageApp
        basename="/model"
        apiBase="/api/model"
        project="bbbbbbbbbbbb"
        onProjectChange={() => {}}
        onProjectsChange={() => {}}
        embedded
      />,
    )
    await waitFor(() => expect(Object.keys(onlyDraft().spec.nodes)).toEqual(['head']))
    first.unmount()
    renderStage({ project: 'bbbbbbbbbbbb' })
    expect(
      await screen.findByRole('region', { name: '保存していない定義の下書き' }),
    ).toBeInTheDocument()
  })

  it('単独では名前と明暗の切り替えを出し, 問題の設定を選ばない', async () => {
    stubApi('/api')
    render(
      <StageApp
        basename=""
        apiBase="/api"
        project={null}
        onProjectChange={() => {}}
        onProjectsChange={() => {}}
        embedded={false}
      />,
    )
    expect(
      await screen.findByRole('heading', { name: 'ML-CreateArchitecture' }),
    ).toBeInTheDocument()
    const dialog = await openProjectDialog()
    await within(dialog).findByRole('button', { name: /総菜の作る量/ })
    expect(within(dialog).queryByRole('region', { name: '選んだ問題の設定' })).toBeNull()
    expect(within(dialog).getByRole('textbox', { name: 'プロジェクトか問題の設定' })).toHaveValue(
      '',
    )
    fireEvent.click(screen.getByRole('button', { name: '暗い表示' }))
    expect(document.documentElement.classList.contains('dark')).toBe(true)
  })
})
