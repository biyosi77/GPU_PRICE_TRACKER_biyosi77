# AI GPU Daily Rental Price Tracker

주요 데이터센터 AI GPU(`H100 SXM`, `H100 PCIe`, `H200`, `B200`, `B300`, `A100 80GB`)의 시간당 임대가격(USD/GPU/hr)과 오픈 마켓플레이스 가용 물량을 매일 자동 수집하여 CSV/JSON 시계열로 누적하고, 전일/7일 대비 변동률을 계산해 텔레그램으로 브리핑하는 파이프라인입니다.

## 데이터 소스
1. **Vast.ai Public API** (`https://console.vast.ai/api/v0/bundles/`): 온디맨드/스팟 실시간 오퍼 및 가용 GPU 수량 집계
2. **RunPod Public GraphQL API** (`https://api.runpod.io/graphql`): Secure Cloud / Community Cloud / 최저 입찰가(`minimumBidPrice`) 조회
3. **Price of Compute Public API** (`https://priceofcompute.com/api/v1/prices/:sku`): 멀티 프로바이더 온디맨드/스팟 중간값 및 7일/30일 히스토리

## 빠른 시작 (GitHub Actions)
1. 본 프로젝트 파일들을 GitHub 저장소 루트에 업로드합니다.
2. GitHub 저장소 **Settings → Secrets and variables → Actions**에서 아래 시크릿을 등록합니다(선택 사항, 미설정 시 데이터 누적 및 리포트 생성만 수행):
   - `TELEGRAM_BOT_TOKEN`: 텔레그램 봇 토큰
   - `TELEGRAM_CHAT_ID`: 브리핑을 받을 채팅방/채널 ID
3. 매일 **한국시간 오전 9시(`UTC 00:00`)**에 자동 실행되며, **Actions → Daily AI GPU Rental Price Tracker → Run workflow**로 언제든 수동 실행할 수 있습니다.

## 로컬 테스트
```bash
pip install -r requirements.txt
python main.py
```
- 네트워크 없는 환경에서 스냅샷 기반 테스트:
```bash
python main.py --seed-file data/daily_gpu_prices.json
```
