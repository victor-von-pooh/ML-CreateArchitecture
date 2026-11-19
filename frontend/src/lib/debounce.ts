import { useEffect, useState } from 'react'

/** 書き換えている途中の値が落ち着いてから使う (確認の回数を抑える) */
export function useDebounced<T>(value: T, delay: number): T {
  const [current, setCurrent] = useState(value)
  useEffect(() => {
    const timer = window.setTimeout(() => setCurrent(value), delay)
    return () => window.clearTimeout(timer)
  }, [value, delay])
  return current
}
