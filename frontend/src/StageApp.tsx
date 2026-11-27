import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import '@xyflow/react/dist/style.css'
import { useMemo, useState } from 'react'

import { App } from './App'
import { ApiContext, ApiError, createApi } from './api/client'
import { StageContext, type StageValue } from './lib/stage'
import { ThemeProvider } from './lib/theme'

/** モデルの作成・選定の段階の画面の根の部品の props (殻と単独で同じ形) */
export interface StageAppProps {
  /** 画面の経路の頭 (例 `/model`). 単独では `""`. このアプリは経路を持たない 1 画面なので使わない */
  basename: string
  /** API の頭 (例 `/api/model`). 単独では `/api` */
  apiBase: string
  /** 殻で選んだ問題の設定の識別子. null は選んでいない */
  project: string | null
  /** 「データセットとプロジェクト」で置き場の別の問題の設定を選んだときに, その識別子で呼ぶ */
  onProjectChange: (project: string | null) => void
  /** 問題の設定の一覧が変わったときに呼ぶ (このアプリは一覧を変えないので呼ばない) */
  onProjectsChange: () => void
  /** 殻の下に載せたか (アプリの名前と明暗の切り替えを出さず, 明暗は読むだけにする) */
  embedded: boolean
}

/** 部品ごとの TanStack Query の client */
function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        refetchOnWindowFocus: false,
        // 誤りの応答などの利用者側のエラーは繰り返さない
        retry: (count, error) => !(error instanceof ApiError && error.status < 500) && count < 1,
      },
    },
  })
}

/**
 * モデルの作成・選定の段階の画面の根の部品
 *
 * 自分の TanStack Query の client を持つ. 経路を持たないのでルーターは持たない. 単独では
 * main.tsx が既定の値 (basename `""`, apiBase `/api`, project null, embedded false) で描く
 */
export function StageApp({ apiBase, project, onProjectChange, embedded }: StageAppProps) {
  const [queryClient] = useState(makeQueryClient)
  const api = useMemo(() => createApi(apiBase), [apiBase])
  const stage = useMemo<StageValue>(
    () => ({ project, onProjectChange, embedded }),
    [project, onProjectChange, embedded],
  )
  return (
    <ThemeProvider readOnly={embedded}>
      <QueryClientProvider client={queryClient}>
        <ApiContext.Provider value={api}>
          <StageContext.Provider value={stage}>
            <App />
          </StageContext.Provider>
        </ApiContext.Provider>
      </QueryClientProvider>
    </ThemeProvider>
  )
}
