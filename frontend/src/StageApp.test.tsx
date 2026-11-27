import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
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
function stubApi(base = '/api/model') {
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
      return Response.json(targetOf(text))
    }
    return new Response(null, { status: 404 })
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
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
    expect(document.documentElement.classList.contains('dark')).toBe(true)
    document.documentElement.classList.remove('dark')
    await waitFor(() => expect(document.documentElement.classList.contains('dark')).toBe(false))
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
