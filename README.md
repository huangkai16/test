# python-aligned-detect

独立 Android 工具，前处理与 YOLO 推理逻辑。

**位置**：python-aligned-detect

## 功能

- **启动即用**：从 `app/src/main/assets/` 自动加载默认模型与测试图，并自动推理（无需手动点选）
- 也可从文件管理器临时选择 **`.tflite` 模型** 与 **测试图片** 做调试
- 前处理与 Python 脚本一致：**直接拉伸 640×640**（非 LetterBox）→ RGB → ÷255 → float32 HWC
- 输出 NMS 后全部检测框，以及 **conf > 0.8** 的高置信框（阈值可在界面调整）

## 内置资源（assets）

将模型与图片放入以下目录，编译进 APK：

```
app/src/main/assets/
  chef.tflite    # 默认模型（YOLOv11 TFLite）
  test.jpg       # 默认测试图（jpg/png/webp 等均可，文件名需与代码一致）
```

默认文件名定义在 `PythonAlignedDetectRunner`：

```kotlin
const val DEFAULT_MODEL_ASSET = "chef.tflite"
const val DEFAULT_IMAGE_ASSET = "test.jpg"
```

更换资源时：替换 `assets` 中对应文件 → 重新编译安装（或 `git add` 后提交推送）即可。若要改文件名，修改上述常量后重编。

> 模型文件可能较大；若 push 失败或仓库变慢，可考虑 [Git LFS](https://git-lfs.com/)。

## 使用流程

### 方式一：assets 内置（推荐，零手动操作）

```bash
cd /Users/huangkai/Documents/zr/python-aligned-detect

# 放入默认资源
cp /path/to/chef.tflite app/src/main/assets/chef.tflite
cp /path/to/test.jpg     app/src/main/assets/test.jpg

./gradlew :app:installDebug   # 需连接 ARM 设备
```

打开 App 后自动：加载模型 → 加载图片 → 推理 → 在「模型输出」卡片展示结果。

界面上的 **「重新加载 assets」** 会重新读取内置资源并再次推理（例如调整 conf/IoU 阈值后想重跑默认样例）。

### 方式二：运行时手动选择

1. 点「选择 .tflite 模型」从文件管理器选模型
2. 点「选择图片」选测试图
3. 点「开始推理」

适用于临时换模型/图片、不想重新编译的场景。

## 构建

```bash
cd /Users/huangkai/Documents/zr/python-aligned-detect
./gradlew :app:assembleDebug
./gradlew :app:installDebug   # 需连接 ARM 设备
```

首次打开请用 Android Studio 同步 Gradle，或确保 `local.properties` 中 `sdk.dir` 指向本机 SDK。

## 与仪器测试对照

| 项目 | 仪器测试 | 本 App |
|------|----------|--------|
| 前处理 | `preprocessAsPythonScript` | 同逻辑 |
| 推理 conf/IoU | 0.3 / 0.5 | 界面可配，默认相同 |
| 高亮阈值 | 0.8 | 默认 0.8 |
| 模型 | APK 内置 | assets 内置，启动自动加载 |
| 测试图 | 仪器测试资源 | assets 内置，启动自动加载 |

结果在界面「模型输出」卡片中展示。

## 目录结构

```
python-aligned-detect/
  app/src/main/
    assets/         # chef.tflite、test.jpg（可提交到 Git）
    java/com/zhongrong/pythonaligned/
      model/        # 前处理、TFLite 检测、Runner
      ui/           # Compose 界面
      viewmodel/    # 启动时 loadBundledAssetsAndRun()
```
