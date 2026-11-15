import { expect, test } from '@playwright/test'

// 保存と読み込みを含むので, 書いた順に実行
test.describe.configure({ mode: 'serial' })

test('例を開くと, 誤り無しで形とパラメータ数が出る', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^cnn_classifier/ }).click()

  // 確認と形の推定の結果がノードと上の欄に出る
  // (誤り 0 件は確認の結果が届く前にも出るので, 例に固有の値を先に待つ)
  await expect(page.locator('[data-id="head"]')).toContainText('[4, 10]', { timeout: 20_000 })
  await expect(page.locator('header')).toContainText('パラメータ数 5.1K')
  await expect(page.locator('header')).toContainText('誤り 0 件')
})

test('部品を置いて選択の欄でつなぎ, 保存して開き直す', async ({ page }) => {
  await page.goto('/')

  // 全結合の層を置いて引数を入れる
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page
    .getByRole('button', { name: /^nn\.Linear/ })
    .first()
    .click()
  await page.getByLabel('in_features').fill('16')
  await page.getByLabel('in_features').press('Enter')
  await page.getByLabel('out_features').fill('4')
  await page.getByLabel('out_features').press('Enter')

  // 入力の口と出力を, 線を引かずに選択の欄でつなぐ
  await page.getByLabel(/^input/).selectOption('x')
  await page.getByRole('button', { name: '出力' }).click()
  await page.getByLabel(/つなぎ先/).selectOption('linear1')
  await expect(page.locator('.react-flow__edge')).toHaveCount(2)
  await expect(page.locator('header')).toContainText('パラメータ数 68', { timeout: 20_000 })
  await expect(page.locator('header')).toContainText('誤り 0 件')

  // 名前を付けて保存すると, 保存していない変更の印が消える
  // (再実行で同じ名前の定義が既にある場合は, 上書きの確認に答える)
  page.on('dialog', (dialog) => void dialog.accept())
  await page.getByLabel('保存する名前').fill('e2e_model')
  await page.getByRole('button', { name: /^保存/ }).click()
  await expect(page.getByRole('button', { name: '保存', exact: true })).toBeVisible()

  // 新しく作ってから開き直すと, 同じ定義
  await page.getByRole('button', { name: '新規' }).click()
  await expect(page.locator('[data-id="linear1"]')).toHaveCount(0)
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^e2e_model \d/ }).click()
  await expect(page.locator('[data-id="linear1"]')).toBeVisible()
  await expect(page.locator('.react-flow__edge')).toHaveCount(2)
})

test('線を選んで Delete で外すと, 誤りが出る', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^e2e_model \d/ }).click()
  await expect(page.locator('.react-flow__edge')).toHaveCount(2)

  // 入力から全結合への線を外すと, 必須の口が未接続の誤り
  // 開いた直後は表示の範囲を合わせる動きで線が動くので, 選べるまで押し直す
  const edge = page.locator('.react-flow__edge').first()
  await expect(async () => {
    await edge.locator('path').first().click({ force: true })
    await expect(edge).toHaveClass(/selected/, { timeout: 1000 })
  }).toPass({ timeout: 10_000 })
  await page.keyboard.press('Delete')
  await expect(page.locator('.react-flow__edge')).toHaveCount(1)
  await expect(page.locator('header')).toContainText('誤り 1 件', { timeout: 20_000 })
})

test('合わない値はその場で理由を出し, ダイアログを閉じると元の所に戻る', async ({ page }) => {
  await page.goto('/')

  // 整数の引数に文字, 範囲の外の数を入れると理由が出て, 直すと消える
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page
    .getByRole('button', { name: /^nn\.Linear/ })
    .first()
    .click()
  const field = page.getByLabel('in_features')
  await field.fill('abc')
  await field.press('Enter')
  await expect(page.getByText('整数を入れてください', { exact: true })).toBeVisible()
  await expect(field).toHaveAttribute('aria-invalid', 'true')
  await field.fill('0')
  await field.press('Enter')
  await expect(page.getByText('1 以上にしてください', { exact: true })).toBeVisible()
  await field.fill('8')
  await field.press('Enter')
  await expect(field).toHaveAttribute('aria-invalid', 'false')

  // ダイアログを閉じるボタンで閉じても, 定義を開いて上の操作が作り直されても,
  // 開いたボタンに戻る
  const open = page.getByRole('button', { name: '開く' })
  await open.click()
  await page.getByRole('dialog').getByRole('button', { name: '閉じる' }).click()
  await expect(page.getByRole('dialog')).toHaveCount(0)
  await expect(open).toBeFocused()
  await page.getByRole('button', { name: 'YAML' }).click()
  await page.getByRole('dialog').getByRole('button', { name: '閉じる' }).click()
  await expect(page.getByRole('button', { name: 'YAML' })).toBeFocused()
  page.once('dialog', (dialog) => void dialog.accept())
  await open.click()
  await page.getByRole('button', { name: /^e2e_model \d/ }).click()
  await expect(page.getByRole('dialog')).toHaveCount(0)
  await expect(page.getByRole('button', { name: '開く' })).toBeFocused()
})

test('元に戻す・やり直すで, 部品の追加と引数の変更を戻せる', async ({ page }) => {
  await page.goto('/')
  const undo = page.getByRole('button', { name: '元に戻す' })
  const redo = page.getByRole('button', { name: 'やり直す' })
  await expect(undo).toBeDisabled()

  // 部品を置いて引数を変え, 引数の変更から順に戻す
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page
    .getByRole('button', { name: /^nn\.Linear/ })
    .first()
    .click()
  await expect(page.locator('[data-id="linear1"]')).toBeVisible()
  await page.waitForTimeout(600)
  await page.getByLabel('in_features').fill('8')
  await page.getByLabel('in_features').press('Enter')
  await page.locator('.react-flow__pane').click({ position: { x: 5, y: 5 } })
  await page.waitForTimeout(600)
  await page.keyboard.press('ControlOrMeta+z')
  await page.locator('[data-id="linear1"]').click()
  await expect(page.getByLabel('in_features')).toHaveValue('')
  await page.locator('.react-flow__pane').click({ position: { x: 5, y: 5 } })
  await undo.click()
  await expect(page.locator('[data-id="linear1"]')).toHaveCount(0)

  // やり直すと部品が戻る
  await redo.click()
  await expect(page.locator('[data-id="linear1"]')).toBeVisible()

  // 欄を通るだけ (値を変えない) では履歴に残らず, やり直しが消えない
  await page.locator('[data-id="linear1"]').click()
  await page.getByLabel('in_features').focus()
  await page.getByLabel('in_features').blur()
  await expect(redo).toBeEnabled()
  await page.locator('.react-flow__pane').click({ position: { x: 5, y: 5 } })
  await page.keyboard.press('ControlOrMeta+Shift+z')
  await page.locator('[data-id="linear1"]').click()
  await expect(page.getByLabel('in_features')).toHaveValue('8')
  // 打ちかけの文字が無い欄の中の Ctrl+Z は定義を戻し, 欄にも出す
  await page.getByLabel('in_features').press('ControlOrMeta+z')
  await expect(page.getByLabel('in_features')).toHaveValue('')
  await expect(page.locator('[data-id="linear1"]')).toBeVisible()
  await expect(redo).toBeEnabled()
  // 打ちかけの文字がある間の Ctrl+Z は, ブラウザの文字の取り消し (定義は戻さない)
  await page.getByLabel('in_features').pressSequentially('3')
  await page.getByLabel('in_features').press('ControlOrMeta+z')
  await expect(page.locator('[data-id="linear1"]')).toBeVisible()
})

test('整列で, つながりの順に左から右へ並べ直し, 元に戻せる', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^residual_mlp/ }).click()
  const box = async (id: string) => (await page.locator(`[data-id="${id}"]`).boundingBox())!
  // 左にあるか (この例は折り返さない長さ. 飛び越すつなぎのために上下には
  // ずれる)
  const before = async (a: string, b: string) => (await box(a)).x < (await box(b)).x

  // 並べ直すと, 入力, つなぎ先の順に右へ
  await page.getByRole('button', { name: '整列' }).click()
  await page.waitForTimeout(400)
  expect(await before('x', 'fc1')).toBe(true)
  expect(await before('fc1', 'act')).toBe(true)
  expect(await before('act', 'add')).toBe(true)
  expect(await before('add', 'head')).toBe(true)
  await page.screenshot({ path: 'test-results/arranged.png' })

  // 元に戻すで, 並べ直す前の位置
  await page.getByRole('button', { name: '元に戻す' }).click()
  await expect(page.getByRole('button', { name: '元に戻す' })).toBeDisabled()
})

test('ノードを引数ごと複製し, 写して貼れる', async ({ page }) => {
  await page.goto('/')
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page
    .getByRole('button', { name: /^nn\.Linear/ })
    .first()
    .click()
  await page.getByLabel('in_features').fill('16')
  await page.getByLabel('in_features').press('Enter')

  // 複製のボタンで, 引数の同じ写しが選ばれる
  await page.getByRole('button', { name: '複製' }).click()
  await expect(page.locator('[data-id="linear2"]')).toBeVisible()
  await expect(page.getByLabel('名前', { exact: true })).toHaveValue('linear2')
  await expect(page.getByLabel('in_features')).toHaveValue('16')

  // Ctrl+C で写し, 何も選ばずに Ctrl+V で貼る. Ctrl+D でも複製
  await page.locator('[data-id="linear1"]').click()
  await page.keyboard.press('ControlOrMeta+c')
  await page.locator('.react-flow__pane').click({ position: { x: 5, y: 5 } })
  await page.keyboard.press('ControlOrMeta+v')
  await expect(page.locator('[data-id="linear3"]')).toBeVisible()
  await page.keyboard.press('ControlOrMeta+d')
  await expect(page.locator('[data-id="linear4"]')).toBeVisible()
  await expect(page.getByLabel('in_features')).toHaveValue('16')
})

test('出力を選んだ後も欄が残らず, 形の推定の失敗を誤りに数え, 上書きは確かめる', async ({
  page,
  request,
}) => {
  await page.goto('/')

  // 出力を選んでから入力を選ぶと, 名前の欄は 1 つ
  await page.getByRole('button', { name: '出力', exact: true }).click()
  await page.getByRole('button', { name: '入力', exact: true }).click()
  await expect(page.getByLabel('名前', { exact: true })).toHaveCount(1)
  await expect(page.getByLabel('名前', { exact: true })).toHaveValue('input1')

  // 形の合わない全結合の層は, 形の推定の失敗として誤りに数える
  await page.getByRole('button', { name: '開く' }).click()
  page.once('dialog', (dialog) => void dialog.accept())
  await page.getByRole('button', { name: /^cnn_classifier/ }).click()
  await page.locator('[data-id="head"]').click()
  await page.getByLabel('in_features').fill('99')
  await page.getByLabel('in_features').press('Enter')
  await expect(page.locator('header')).toContainText('誤り 1 件', { timeout: 20_000 })
  await page.locator('.react-flow__pane').click({ position: { x: 5, y: 5 } })
  await expect(page.getByRole('region', { name: '形の推定で失敗したもの' })).toContainText('head')

  // 保存した定義 (前のテストで保存した e2e_model) と同じ名前で保存する
  // 場合は確かめ, 断れば保存しない
  const before = await (await request.get('/api/models/e2e_model')).text()
  await page.getByLabel('保存する名前').fill('e2e_model')
  let asked = ''
  page.once('dialog', (dialog) => {
    asked = dialog.message()
    void dialog.dismiss()
  })
  await page.getByRole('button', { name: /^保存/ }).click()
  await expect.poll(() => asked).toContain('e2e_model')
  await page.waitForTimeout(500)
  expect(await (await request.get('/api/models/e2e_model')).text()).toBe(before)
  await expect(page.getByRole('button', { name: '保存 *' })).toBeVisible()
})

test('狭い画面では上の操作をアイコンにし, ミニマップを出さない', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 })
  await page.goto('/')
  // 文字は読み上げの名前として残り, 上の操作は 1 行 (高さが 1 行分)
  for (const name of ['整列', '入力', '出力', 'YAML', 'Python']) {
    await expect(page.getByRole('button', { name, exact: true })).toBeVisible()
  }
  expect((await page.locator('header').boundingBox())!.height).toBeLessThan(60)
  // 新規の直後 (部品が無い) は, 誤りの数ではなく始め方
  await expect(page.locator('header')).toContainText('部品なし')
  await expect(page.locator('.react-flow__minimap')).toBeHidden()
})

test('保存した定義を開くの一覧から消せる', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く' }).click()
  const dialog = page.getByRole('dialog')
  await expect(dialog.getByRole('button', { name: /^e2e_model \d/ })).toBeVisible()

  // 確かめを断ると残り, 受けると一覧から外れる
  page.once('dialog', (confirm) => void confirm.dismiss())
  await dialog.getByRole('button', { name: 'e2e_model を消す' }).click()
  await expect(dialog.getByRole('button', { name: /^e2e_model \d/ })).toBeVisible()
  page.once('dialog', (confirm) => void confirm.accept())
  await dialog.getByRole('button', { name: 'e2e_model を消す' }).click()
  await expect(dialog.getByRole('button', { name: /^e2e_model \d/ })).toHaveCount(0)
})

test('組み立ての途中 (出力が無い, つないでいないノードがある) でも形を出す', async ({ page }) => {
  await page.goto('/')

  // 出力を置く前でも, つないだノードには形が出る
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page
    .getByRole('button', { name: /^nn\.Linear/ })
    .first()
    .click()
  await page.getByLabel('in_features').fill('16')
  await page.getByLabel('in_features').press('Enter')
  await page.getByLabel('out_features').fill('4')
  await page.getByLabel('out_features').press('Enter')
  await page.getByLabel(/^input/).selectOption('x')
  await expect(page.locator('[data-id="linear1"]')).toContainText('[4, 4]', { timeout: 20_000 })

  // 複製したつないでいないノードがあっても, 他のノードの形は消えない
  await page.getByRole('button', { name: '複製' }).click()
  await expect(page.locator('[data-id="linear2"]')).toBeVisible()
  await page.waitForTimeout(1500)
  await expect(page.locator('[data-id="linear1"]')).toContainText('[4, 4]')
  await expect(page.locator('header')).toContainText('パラメータ数 68')
})

test('複数つなぐ口のつなぎ先を並べ替えられる', async ({ page }) => {
  await page.goto('/')
  await page.getByPlaceholder('部品を探す').fill('op.Concat')
  await page
    .getByRole('button', { name: /^op\.Concat/ })
    .first()
    .click()
  await page.getByPlaceholder('部品を探す').fill('nn.ReLU')
  await page
    .getByRole('button', { name: /^nn\.ReLU / })
    .first()
    .click()
  await page.getByLabel(/^input/).selectOption('x')

  // concat1 に x と relu1 の順でつなぎ, relu1 を上へ
  await page.locator('[data-id="concat1"]').click()
  await page.getByLabel('xs につなぎ先を加える').selectOption('x')
  await page.getByLabel('xs につなぎ先を加える').selectOption('relu1')
  await page.getByRole('button', { name: 'xs の 2 番目のつなぎ先 relu1 を上へ' }).click()
  // 動かした先の行のボタンにフォーカスが移る (先頭なので下へ)
  await expect(
    page.getByRole('button', { name: 'xs の 1 番目のつなぎ先 relu1 を下へ' }),
  ).toBeFocused()
  await expect(
    page.getByRole('button', { name: 'xs の 1 番目のつなぎ先 relu1 を外す' }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'xs の 2 番目のつなぎ先 x を外す' })).toBeVisible()
})

test('大きすぎて形を推定しない場合は, 上の欄で知らせる', async ({ page }) => {
  await page.goto('/')
  await page.locator('[data-id="x"]').click()
  await page.getByLabel('shape').fill('[100000, 100000]')
  await page.getByLabel('shape').press('Enter')
  await expect(page.locator('header')).toContainText('形は未推定', { timeout: 20_000 })
  await page.getByLabel('shape').fill('[4, 16]')
  await page.getByLabel('shape').press('Enter')
  await expect(page.locator('header')).not.toContainText('形は未推定', { timeout: 20_000 })
})

test('キーボードだけで部品を置き, 名前を変えて戻し, Ctrl+S で保存する', async ({ page }) => {
  await page.goto('/')

  // 検索欄の Enter で一番上の部品を置き, 詳細の最初の欄に移る
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page.getByPlaceholder('部品を探す').press('Enter')
  await expect(page.locator('[data-id="linear1"]')).toBeVisible()
  await expect(page.getByLabel('名前', { exact: true })).toBeFocused()

  // 名前を Enter で変えても名前の欄に残り, 元に戻すと元の名前を選んだまま
  await page.getByLabel('名前', { exact: true }).fill('fc')
  await page.getByLabel('名前', { exact: true }).press('Enter')
  await expect(page.getByLabel('名前', { exact: true })).toBeFocused()
  await expect(page.getByLabel('名前', { exact: true })).toHaveValue('fc')
  await page.locator('.react-flow__pane').click({ position: { x: 5, y: 5 } })
  await page.locator('[data-id="fc"]').click()
  await page.keyboard.press('ControlOrMeta+z')
  await expect(page.getByLabel('名前', { exact: true })).toHaveValue('linear1')

  // 保存する名前の欄の中でも Ctrl+S で保存する (保存していない変更の印が消える)
  await page.getByLabel('保存する名前').fill('keyboard_model')
  await page.getByLabel('保存する名前').press('ControlOrMeta+s')
  await expect(page.getByRole('button', { name: '保存', exact: true })).toBeVisible()
})

test('保存した定義を部分モデルとして写し, 部品として置いてつなぐ', async ({ page }) => {
  await page.goto('/')

  // 同梱の例を部分モデルとして写すと, 部品の一覧の「部分モデル」に出る
  await page.getByRole('button', { name: '別の定義を部品にする' }).click()
  await page.getByRole('button', { name: /^residual_mlp/ }).click()
  const unit = page.getByRole('button', { name: /^sub\.residual_mlp/ })
  await expect(unit).toBeVisible()

  // 置いて入力と出力をつなぐと, 部分モデルの出力の形とパラメータ数が出る
  await unit.click()
  await page.getByLabel(/^x/).selectOption('x')
  await page.getByRole('button', { name: '出力', exact: true }).click()
  await page.getByLabel(/つなぎ先/).selectOption('residual_mlp1')
  await expect(page.locator('[data-id="residual_mlp1"]')).toContainText('[4, 3]', {
    timeout: 20_000,
  })
  await expect(page.locator('header')).toContainText('誤り 0 件')

  // 使っている部分モデルは消せず, ノードを消すと消せる
  await page.getByRole('button', { name: '部分モデル sub.residual_mlp を消す' }).click()
  await expect(page.getByRole('status').filter({ hasText: '消せません' })).toBeVisible()
  await expect(unit).toBeVisible()

  // 中身は詳細で読める
  await page.locator('[data-id="residual_mlp1"]').click()
  await page.getByText('部分モデルの中身', { exact: true }).click()
  await expect(page.locator('aside[aria-label="詳細"]')).toContainText('fc1: nn.Linear')
  await page.locator('[data-id="residual_mlp1"]').click()
  await page.keyboard.press('Delete')
  await page.getByRole('button', { name: '部分モデル sub.residual_mlp を消す' }).click()
  await expect(unit).toHaveCount(0)
  await expect(page.getByRole('status').filter({ hasText: '消しました' })).toBeVisible()
})

test('誤りのある定義を部分モデルにする前に確かめる', async ({ page }) => {
  // 必須の引数とつなぎ先の無い層だけの定義を保存
  await page.goto('/')
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page
    .getByRole('button', { name: /^nn\.Linear/ })
    .first()
    .click()
  await page.getByLabel('保存する名前').fill('e2e_broken')
  await page.getByRole('button', { name: /^保存 \*/ }).click()
  await expect(page.getByRole('button', { name: '保存', exact: true })).toBeVisible()
  await page.getByRole('button', { name: '新規' }).click()

  // 写すかを確かめ, 断ると写さない
  let asked = ''
  page.once('dialog', (dialog) => {
    asked = dialog.message()
    void dialog.dismiss()
  })
  await page.getByRole('button', { name: '別の定義を部品にする' }).click()
  await page.getByRole('button', { name: /^e2e_broken/ }).click()
  await expect.poll(() => asked).toContain('誤りが')
  await page.getByRole('button', { name: '閉じる' }).click()
  await expect(page.getByRole('button', { name: /^sub\.e2e_broken/ })).toHaveCount(0)
})

test('部分モデルの中身を編集し, 本体に戻ると使うノードの形が変わる', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '別の定義を部品にする' }).click()
  await page.getByRole('button', { name: /^residual_mlp/ }).click()
  await page.getByRole('button', { name: /^sub\.residual_mlp/ }).click()
  await page.getByLabel(/^x/).selectOption('x')
  await page.getByRole('button', { name: '出力', exact: true }).click()
  await page.getByLabel(/つなぎ先/).selectOption('residual_mlp1')
  await expect(page.locator('[data-id="residual_mlp1"]')).toContainText('[4, 3]', {
    timeout: 20_000,
  })

  // 中身を編集して, 最後の層の出力を 5 にする
  await page.locator('[data-id="residual_mlp1"]').click()
  await page.getByRole('button', { name: '部分モデルの中身を編集' }).click()
  await expect(page.getByRole('complementary', { name: '編集している部分モデル' })).toContainText(
    'sub.residual_mlp',
  )
  // 編集している部分モデル自身は部品の一覧に出ない (循環)
  await expect(page.getByRole('button', { name: /^sub\.residual_mlp/ })).toHaveCount(0)
  await page.locator('[data-id="head"]').click()
  await page.getByLabel('out_features').fill('5')
  await page.getByLabel('out_features').press('Enter')
  // 中の形は, 本体で使う所に入る形 (例の [8, 16] ではなく本体の [4, 16]) で推定
  await expect(page.locator('[data-id="head"]')).toContainText('[4, 5]', { timeout: 20_000 })
  // YAML は書いたままの入力の形 (推定に使った本体の形ではない)
  await page.getByRole('button', { name: 'YAML', exact: true }).click()
  await expect(page.getByRole('dialog')).toContainText('shape: [8, 16]')
  await page.getByRole('button', { name: '閉じる' }).click()
  await expect(page.getByRole('complementary', { name: '編集している部分モデル' })).toContainText(
    'ノード residual_mlp1 に入る形',
  )

  // 本体に戻ると, 部分モデルを使うノードの形が変わり, 元に戻すで戻る
  await page.getByRole('button', { name: '本体', exact: true }).click()
  await expect(page.locator('[data-id="residual_mlp1"]')).toContainText('[4, 5]', {
    timeout: 20_000,
  })
  await page.keyboard.press('ControlOrMeta+z')
  await expect(page.locator('[data-id="residual_mlp1"]')).toContainText('[4, 3]', {
    timeout: 20_000,
  })
})

test('元の定義を直した後に, 部分モデルを写し直せる', async ({ page }) => {
  page.on('dialog', (dialog) => void dialog.accept())
  // 写す元の定義 (x → 全結合 16 → 4)
  await page.goto('/')
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page
    .getByRole('button', { name: /^nn\.Linear/ })
    .first()
    .click()
  await page.getByLabel('in_features').fill('16')
  await page.getByLabel('in_features').press('Enter')
  await page.getByLabel('out_features').fill('4')
  await page.getByLabel('out_features').press('Enter')
  await page.getByLabel(/^input/).selectOption('x')
  await page.getByRole('button', { name: '出力', exact: true }).click()
  await page.getByLabel(/つなぎ先/).selectOption('linear1')
  await page.getByLabel('保存する名前').fill('e2e_src')
  await page.getByRole('button', { name: /^保存 \*/ }).click()
  await expect(page.getByRole('button', { name: '保存', exact: true })).toBeVisible()

  // 写して使う定義
  await page.getByRole('button', { name: '新規' }).click()
  await page.getByRole('button', { name: '別の定義を部品にする' }).click()
  await page.getByRole('button', { name: /^e2e_src/ }).click()
  await page.getByRole('button', { name: /^sub\.e2e_src/ }).click()
  await page.getByLabel(/^x/).selectOption('x')
  await page.getByRole('button', { name: '出力', exact: true }).click()
  await page.getByLabel(/つなぎ先/).selectOption('e2e_src1')
  await expect(page.locator('[data-id="e2e_src1"]')).toContainText('[4, 4]', { timeout: 20_000 })
  await page.getByLabel('保存する名前').fill('e2e_user')
  await page.getByRole('button', { name: /^保存 \*/ }).click()
  await expect(page.getByRole('button', { name: '保存', exact: true })).toBeVisible()

  // 元の定義の出力を 8 にして保存
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^e2e_src \d/ }).click()
  await page.locator('[data-id="linear1"]').click()
  await page.getByLabel('out_features').fill('8')
  await page.getByLabel('out_features').press('Enter')
  await page.getByRole('button', { name: /^保存 \*/ }).click()
  await expect(page.getByRole('button', { name: '保存', exact: true })).toBeVisible()

  // 使う定義を開いて写し直すと, 形が変わる
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^e2e_user \d/ }).click()
  await expect(page.locator('[data-id="e2e_src1"]')).toContainText('[4, 4]', { timeout: 20_000 })
  await page.getByRole('button', { name: '部分モデル sub.e2e_src を元の定義から写し直す' }).click()
  await expect(page.locator('[data-id="e2e_src1"]')).toContainText('[4, 8]', { timeout: 20_000 })
})

test('選んでいるものから新しい部品へ線を引き, 出力はつながっていないノードにつなぐ', async ({
  page,
}) => {
  await page.goto('/')
  await page.locator('[data-id="x"]').click()
  await page.getByPlaceholder('部品を探す').fill('nn.ReLU')
  await page.getByPlaceholder('部品を探す').press('Enter')
  await expect(page.locator('.react-flow__edge')).toHaveCount(1)
  await page.getByRole('button', { name: '出力', exact: true }).click()
  await expect(page.locator('.react-flow__edge')).toHaveCount(2)
  await expect(page.locator('header')).toContainText('誤り 0 件', { timeout: 20_000 })

  // 詳細に入る形と出る形が出る
  await page.locator('[data-id="relu1"]').click()
  await expect(page.locator('aside[aria-label="詳細"]')).toContainText('input ← x: [4, 16]')
})

test('口の多いノードも幅を超えず, 例を開いたときにノードが重ならない', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^transformer_encoder/ }).click()
  await expect(page.locator('[data-id="layer1"]')).toContainText('[', { timeout: 20_000 })
  const boxes = await page.locator('.react-flow__node').evaluateAll((nodes) =>
    nodes.map((node) => {
      const rect = node.getBoundingClientRect()
      return { x: rect.x, y: rect.y, right: rect.right, bottom: rect.bottom }
    }),
  )
  for (const [index, a] of boxes.entries()) {
    for (const b of boxes.slice(index + 1)) {
      const overlap = a.x < b.right && b.x < a.right && a.y < b.bottom && b.y < a.bottom
      expect(overlap).toBe(false)
    }
  }
})

test('部分モデルを 1 つのノードだけのものにし, Esc で本体に戻り, / で部品を探す', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^stacked_blocks/ }).click()
  await expect(page.locator('[data-id="block2"]')).toContainText('[8, 64]', { timeout: 20_000 })

  // block2 だけの部分モデルにして中身を変えても, 他のブロックは変わらない
  await page.locator('[data-id="block2"]').click()
  await page.getByRole('button', { name: 'このノードだけの部分モデルにする' }).click()
  await expect(page.locator('[data-id="block2"]')).toContainText('sub.res_block_block2')
  await page.getByRole('button', { name: '部分モデルの中身を編集' }).click()
  // Esc で本体に戻る (選んでいるものの選択も外れる)
  await page.locator('[data-id="act"]').click()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('complementary', { name: '編集している部分モデル' })).toHaveCount(0)
  // 1 つのノードだけが使う部分モデルには, 写す操作を出さない
  await page.locator('[data-id="block2"]').click()
  await expect(page.getByRole('button', { name: 'このノードだけの部分モデルにする' })).toHaveCount(
    0,
  )
  await expect(page.locator('[data-id="block1"]')).toContainText('sub.res_block')

  // / で部品を探す欄に移る
  await page.locator('.react-flow__pane').click({ position: { x: 5, y: 5 } })
  await page.keyboard.press('/')
  await expect(page.getByPlaceholder('部品を探す')).toBeFocused()
})

test('保存すると知らせが出て, YAML と Python のコードをファイルとして保存できる', async ({
  page,
}) => {
  page.on('dialog', (dialog) => void dialog.accept())
  await page.goto('/')
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^mlp/ }).click()
  // 開いた定義の名前で上の操作を作り直してから名前を変える
  await expect(page.getByLabel('保存する名前')).toHaveValue('mlp')
  await page.getByLabel('保存する名前').fill('e2e_download')
  await page.getByRole('button', { name: /^保存/ }).click()
  await expect(
    page.getByRole('status').filter({ hasText: 'e2e_download に保存しました' }),
  ).toBeVisible()

  await page.getByRole('button', { name: 'Python', exact: true }).click()
  const python = page.waitForEvent('download')
  await page.getByRole('button', { name: 'e2e_download.py として保存' }).click()
  expect((await python).suggestedFilename()).toBe('e2e_download.py')
  await page.getByRole('button', { name: '閉じる' }).click()

  await page.getByRole('button', { name: 'YAML', exact: true }).click()
  const yaml = page.waitForEvent('download')
  await page.getByRole('button', { name: 'e2e_download.yaml として保存' }).click()
  expect((await yaml).suggestedFilename()).toBe('e2e_download.yaml')
})

test('部品の一覧からキャンバスにドラッグして置ける', async ({ page }) => {
  await page.goto('/')
  await page.getByPlaceholder('部品を探す').fill('nn.ReLU')
  const source = page.getByRole('button', { name: /^nn\.ReLU/ }).first()
  await source.dragTo(page.locator('.react-flow__pane'), { targetPosition: { x: 500, y: 400 } })
  await expect(page.locator('[data-id="relu1"]')).toBeVisible()
  // ドラッグで置いた場合は線を引かない
  await expect(page.locator('.react-flow__edge')).toHaveCount(0)
})

test('狭い画面では左右の欄を畳んで始まり, 開いて使える', async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 768 })
  await page.goto('/')
  await expect(page.locator('[data-id="x"]')).toBeVisible()
  await expect(page.getByRole('complementary', { name: '部品の一覧' })).toBeHidden()
  await expect(page.getByRole('complementary', { name: '詳細' })).toBeHidden()

  // / で部品の一覧を開いて検索に移り, Enter で置くと詳細も開く
  await page.keyboard.press('/')
  await expect(page.getByPlaceholder('部品を探す')).toBeFocused()
  await page.getByPlaceholder('部品を探す').fill('nn.ReLU')
  await page.getByPlaceholder('部品を探す').press('Enter')
  await expect(page.getByRole('complementary', { name: '詳細' })).toBeVisible()

  // 畳むボタンで閉じる
  await page.getByRole('button', { name: '部品の一覧を畳む' }).click()
  await expect(page.getByRole('complementary', { name: '部品の一覧' })).toBeHidden()
})

test('部品のノードの出力の口から線を引け, Delete で消すと 1 回で元に戻る', async ({ page }) => {
  await page.goto('/')
  // x → relu1 (線を引いて置く), 何も選ばずに relu2 を置く
  await page.locator('[data-id="x"]').click()
  await page.getByPlaceholder('部品を探す').fill('nn.ReLU')
  await page.getByPlaceholder('部品を探す').press('Enter')
  await page.locator('.react-flow__pane').click({ position: { x: 5, y: 5 } })
  await page.getByPlaceholder('部品を探す').press('Enter')
  await expect(page.locator('.react-flow__edge')).toHaveCount(1)

  // relu1 の出力の口から relu2 の入力の口へドラッグ
  const from = page.locator('[data-id="relu1"] [data-handleid="out:out"]')
  const to = page.locator('[data-id="relu2"] [data-handleid="in:input"]')
  await from.dragTo(to)
  await expect(page.locator('.react-flow__edge')).toHaveCount(2)

  // relu1 を Delete で消し, 1 回の元に戻すで線ごと戻る
  await page.locator('[data-id="relu1"]').click()
  await page.keyboard.press('Delete')
  await expect(page.locator('[data-id="relu1"]')).toHaveCount(0)
  await page.keyboard.press('ControlOrMeta+z')
  await expect(page.locator('[data-id="relu1"]')).toBeVisible()
  await expect(page.locator('.react-flow__edge')).toHaveCount(2)
})

test('注意のあるノードを全体の情報に出し, 別名で部品を探せる', async ({ page }) => {
  await page.goto('/')
  // 出力につながない層は注意 (選ばなくても全体の情報に出る)
  await page.locator('[data-id="x"]').click()
  // (別名 residual で op.Add. つなぎ先が 1 つだけの注意も出る)
  await page.getByPlaceholder('部品を探す').fill('residual')
  await page.getByPlaceholder('部品を探す').press('Enter')
  await expect(page.locator('[data-id="add1"]')).toBeVisible()
  await page.locator('.react-flow__pane').click({ position: { x: 5, y: 5 } })
  const warnings = page.getByRole('region', { name: '注意のあるノード' })
  await expect(warnings).toContainText('add1', { timeout: 20_000 })
  await expect(warnings).toContainText('つなぎ先が 1 つだけ')
})

test('値を反映した後の欄の中の Ctrl+Z は, 定義を戻して欄にも出す', async ({ page }) => {
  await page.goto('/')
  await page.locator('[data-id="x"]').click()
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page.getByPlaceholder('部品を探す').press('Enter')
  await page.getByLabel('in_features').fill('16')
  await page.getByLabel('in_features').press('Tab')
  await page.getByLabel('out_features').fill('4')
  await page.getByLabel('out_features').press('Enter')
  await expect(page.locator('[data-id="linear1"]')).toContainText('[4, 4]', { timeout: 20_000 })

  // out_features の欄にいたまま Ctrl+Z で, 定義も欄も 4 の前 (空) に戻る
  await page.keyboard.press('ControlOrMeta+z')
  await expect(page.getByLabel('out_features')).toHaveValue('')
  await expect(page.locator('[data-id="linear1"]')).not.toContainText('[4, 4]', {
    timeout: 20_000,
  })
  // やり直すと戻る
  await page.keyboard.press('ControlOrMeta+Shift+z')
  await expect(page.getByLabel('out_features')).toHaveValue('4')
})

test('キャンバスへ移るリンク, 欄の Esc でノードに戻る, ノードと線の読み上げの名前', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page.locator('[data-id="x"]')).toBeVisible()
  await page.keyboard.press('Tab')
  await expect(page.getByRole('link', { name: 'キャンバスへ移る' })).toBeFocused()

  await page.locator('[data-id="x"]').click()
  await page.getByPlaceholder('部品を探す').fill('nn.ReLU')
  await page.getByPlaceholder('部品を探す').press('Enter')
  // キーボードで置くと詳細の最初の欄に移り, Esc でノードに戻る
  await expect(page.getByLabel('名前', { exact: true })).toBeFocused()
  await page.keyboard.press('Escape')
  await expect(page.locator('[data-id="relu1"]')).toBeFocused()
  await expect(page.locator('[data-id="relu1"]')).toHaveAttribute(
    'aria-label',
    /ノード relu1 \(nn\.ReLU\)/,
  )
  await expect(page.locator('.react-flow__edge').first()).toHaveAttribute(
    'aria-label',
    'x から relu1 の input への線',
  )
})
