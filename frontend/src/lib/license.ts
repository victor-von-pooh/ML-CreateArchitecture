// 重みの出どころを決める引数 (これと重みを読む引数だけが変わったときに聞き直す)
const SOURCE_PARAMS = ['model_name', 'revision', 'name', 'encoder_name']

/** ライセンスに関わる引数だけを, 決まった順の JSON の文字にする (落ち着くまで待つ値を安定させる) */
export function licenseParams(params: Record<string, unknown>, weightsParams: string[]): string {
  const keys = [...new Set([...SOURCE_PARAMS, ...weightsParams])].sort()
  return JSON.stringify(keys.filter((key) => key in params).map((key) => [key, params[key]]))
}
