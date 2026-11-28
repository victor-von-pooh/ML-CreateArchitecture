import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError, contractDetail, request } from './client'

/** fetch を, 決まった状態と JSON の中身を返すものに差し替える */
function mockFetch(status: number, body: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn(() =>
      Promise.resolve(new Response(JSON.stringify(body), { status, statusText: 'Conflict' })),
    ),
  )
}

/** request が投げた例外を返す */
async function caught(): Promise<unknown> {
  try {
    await request('/api/project/model', { method: 'PUT' })
  } catch (error) {
    return error
  }
  throw new Error('例外が投げられませんでした')
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('readError と contractDetail', () => {
  it('文の detail はその文で, 中身を持たない', async () => {
    mockFetch(409, { detail: '/p/models/model.yaml が既にあります' })
    const error = await caught()
    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).message).toBe('/p/models/model.yaml が既にあります')
    expect((error as ApiError).detail).toBeNull()
    expect(contractDetail(error)).toBeNull()
  })

  it('骨組みの約束の detail は, 文と直す場所と知らせを返す', async () => {
    mockFetch(409, {
      detail: {
        message: '合わない所が 2 件あります',
        findings: [
          { level: 'error', id: 'inputs.x', message: '入力 x の最後の大きさを 3 にしてください' },
          { id: 5 },
        ],
        notes: ['確かめていません'],
        widths: { features: 3 },
      },
    })
    const error = await caught()
    expect((error as ApiError).message).toBe('合わない所が 2 件あります')
    expect(contractDetail(error)).toEqual({
      message: '合わない所が 2 件あります',
      findings: [
        { level: 'error', id: 'inputs.x', message: '入力 x の最後の大きさを 3 にしてください' },
        // 崩れた項目も文字にして出す
        { level: 'error', id: '5', message: '' },
      ],
      notes: ['確かめていません'],
    })
  })

  it('入力の検証の誤りは欄ごとに 1 行で, 409 でなければ約束の detail としない', async () => {
    mockFetch(422, {
      detail: [
        { loc: ['body', 'spec', 'nodes'], msg: 'Field required' },
        { loc: ['query', 'path'], msg: 'bad' },
      ],
    })
    const error = await caught()
    expect((error as ApiError).message).toBe('spec.nodes: Field required\npath: bad')
    expect(contractDetail(error)).toBeNull()
    mockFetch(422, { detail: { message: '断りました', findings: [] } })
    expect(contractDetail(await caught())).toBeNull()
  })

  it('既定の 404 は経路の誤りで, JSON でない本文は状態の番号と文', async () => {
    mockFetch(404, { detail: 'Not Found' })
    expect(((await caught()) as ApiError).message).toBe('API の経路が見つかりません (404)')
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve(new Response('oops', { status: 502, statusText: 'Bad Gateway' })),
      ),
    )
    expect(((await caught()) as ApiError).message).toBe('502 Bad Gateway')
  })
})
