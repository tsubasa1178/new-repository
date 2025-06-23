# new-repository
# Docker最初のサンプル

実行コマンド

```bash
docker build -t mypython:0.1 .
docker run -it --rm -v ${PWD}:/mnt mypython:0.1 python hello.py


#### 実行コマンド
```bash
git add README.md
git commit -m "docs: 初期READMEを追加"

```bash
docker compose up -d
docker compose exec mypython python hello.py
docker compose down

#### 実行コマンド
```bash
git add README.md
git commit -m "docs: docker-composeによる実行手順をREADMEに追記"

## 補足

- `hello.py` は /mnt にマウントされているコードを実行します。
- `mypython:0.1` はカスタムビルドされたPython環境です。
#### 実行コマンド
git add README.md
git commit -m "docs: hello.pyの実行方法と補足説明をREADMEに追加"