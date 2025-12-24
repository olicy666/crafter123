专门针对在 ViewCrafter123 中实现**无需训练、即插即用的位姿/轨迹自动生成（方案 C）**，从而不再手工写 `traj_txt`。
文档里的关键接口、数据格式、插值行为、以及 ViewCrafter 现有代码的挂载点，都基于仓库现状来写：

* `single_view_txt` 的 `traj_txt` 格式是 **三行序列**：`d_phi` / `d_theta` / `d_r`，每行从 0 开始，长度 2~25；程序会做平滑插值并渲染轨迹可视化视频。([GitHub][1])
* `viewcrafter.py` 中 `single_view_txt` 读取 `traj_txt` 后调用 `generate_traj_txt(...)` 生成 Pytorch3D cameras，再渲染、再扩散。([GitHub][2])
* `generate_traj_txt` 会把 `r` 序列按 anchor 相机的 `c2ws_anchor[0,2,3]` 进行缩放（也就是“以中心深度为半径尺度”）。([GitHub][3])
* 现有 `run_render(..., nbv=True)` 的路径里，`ViewCrafter.run_render` **硬编码了 `nbv=False` 传给 `render_pcd`**，这会让“基于 mask 的视点选择”逻辑无法真正拿到 `viewmask`，属于需要顺手修的关键 bug。([GitHub][2])

---

## ViewCrafter123-AutoTraj

### 目标

在 **不训练** ViewCrafter123 / 不改扩散模型权重 的前提下，实现一个“自动轨迹规划器”：

* 输入：单张参考图（与现有 `--image_dir` 一致）
* 输出：自动生成的 `traj_txt`（或内部序列），并直接驱动 ViewCrafter 的 `single_view_txt` 渲染+扩散流程
* 效果：用户只需选择“运动风格/探索强度”，不再手工写 `loop1.txt` 这类轨迹文件

核心思路（方案 C）：
**利用 ViewCrafter123 已有的点云渲染器 + 可见性 mask（viewmask）作为“信息增益/遮挡暴露”打分**，做一个**训练无关**的相机轨迹搜索/规划器；再把规划结果转换成 ViewCrafter 原生 `traj_txt`（`d_phi/d_theta/d_r` 序列）喂回去。

---

## 1. ViewCrafter123 当前轨迹输入机制（必须对齐）

### 1.1 两种主要模式

* `--mode single_view_target`：输入一个目标增量 `(d_theta, d_phi, d_r)`（以及可选平移 `d_x, d_y`），内部线性插值生成 25 帧相机轨迹。([GitHub][1])
* `--mode single_view_txt`：输入 `--traj_txt path/to.txt`，文件三行分别为 `d_phi` / `d_theta` / `d_r` 序列，长度 2~25，每行从 0 开始；内部对每条序列做平滑/线性插值生成 25 帧轨迹，并可输出 `viz_traj.mp4`。([GitHub][1])

### 1.2 `traj_txt` 精确格式（三行）

* 第 1 行：`d_phi` 序列（单位：度）
* 第 2 行：`d_theta` 序列（单位：度）
* 第 3 行：`d_r` 序列（单位：比例，后续会乘以 anchor 相机深度尺度）
  约束：
* 每行序列都必须从 0 开始
* 每行长度 ∈ [2, 25]
* 相邻差分不要过大，避免镜头突变导致生成漂移/伪影（官方 tip）。([GitHub][1])

### 1.3 `generate_traj_txt` 的插值与尺度（必须复用）

`utils/pvd_utils.py::generate_traj_txt(...)` 行为要点：

* 若序列长度 > 3，用 `txt_interpolation(..., mode='smooth')`；否则用 `linear`。([GitHub][3])
* `r` 序列会乘以 `c2ws_anchor[0,2,3]`（anchor 相机位姿的 z 平移，等价于 ViewCrafter 里“中心像素深度决定球半径”这一套）。([GitHub][3])
* 每帧把 `(theta, phi, r)` 送进 `sphere2pose(...)` 得到新的 `c2w`，然后转换到 Pytorch3D cameras。([GitHub][3])

---

## 2. 方案 C：Mask-aware 自动轨迹规划器（训练无关）

### 2.1 直觉

你现在缺的是“位姿输入的自动化”。ViewCrafter 已经能：

1. 用 DUSt3R 得到稠密点云/深度/相机；([GitHub][2])
2. 给定相机轨迹渲染点云得到 `render_results`；([GitHub][2])
3. 再用视频扩散做新视角视频。([GitHub][2])

所以我们做的改进应该集中在：
**在渲染阶段之前，自动产生一条“值得探索”的相机轨迹**（而不是让用户手工写三行序列）。

### 2.2 可落地的无训练规划指标：`viewmask` 打分

`render_pcd(..., nbv=True)` 会额外渲染一个 “mask pointcloud”，得到 `view_masks`（本质是可见性/覆盖度图）。([GitHub][2])

规划策略（默认建议）：

* 生成多个候选 keyframe（候选的 `d_phi/d_theta/d_r`），把它们变成 cameras
* 先**只渲染 mask**（不用扩散），得到每个候选的 `viewmask`
* 用 `score = viewmask.sum()` 或 `score = visible_ratio` 作为“当前点云在该视角下的覆盖度”
* 选择**覆盖度更低**的视角（即更大空洞/遮挡），用来驱动扩散去“补全看不到的区域”，进而扩展点云（这是 ViewCrafter 论文摘要里提到的“自动揭示并处理遮挡”的方向）。([arXiv][4])

> 注意：你如果希望“更稳、更少漂移”，也可以反过来选覆盖度更高的视角（保证重叠），这作为可切换策略实现即可。

---

## 3. 必须先修的关键 bug（否则方案 C 没意义）

### 3.1 问题

`ViewCrafter.run_render(..., nbv=False)` 虽然提供了 `nbv` 参数，但内部调用 `render_pcd` 时**写死了 `nbv=False`**，导致无论外面传什么都拿不到 `viewmask`。([GitHub][2])

### 3.2 修复

文件：`viewcrafter.py`
函数：`run_render(self, pcd, imgs, masks, H, W, camera_traj, num_views, nbv=False)`

把这一段：

```python
render_results, viewmask = self.render_pcd(pcd, imgs, masks, num_views, renderer, self.device, nbv=False)
```

改为：

```python
render_results, viewmask = self.render_pcd(pcd, imgs, masks, num_views, renderer, self.device, nbv=nbv)
```

修复后，你才能在“候选视角评估”阶段拿到有效的 `viewmask`。

---

## 4. 代码改动总览（建议按模块化做）

### 4.1 新增文件

新增：`utils/auto_traj_planner.py`

提供：

* 候选 keyframe 采样
* 候选评分（mask-based）
* keyframe 序列到 `traj_txt`（三行）导出
* （可选）把规划结果直接返回给 `generate_traj_txt`，不落盘也可

### 4.2 修改文件

1. `viewcrafter.py`

* 修 bug：`run_render` 透传 `nbv`
* 新增模式：`single_view_autotraj`（或你喜欢的名字）
* 在 `nvs_single_view()` 分支里加入自动规划逻辑（生成 phi/theta/r 序列后复用 `generate_traj_txt`）

2. `configs/infer_config.py`

* `--mode` help 文案增加 `single_view_autotraj`
* 新增 planner 参数（见下）

---

## 5. 新模式：`single_view_autotraj` 的端到端数据流

### 5.1 输入

与现有一致：

* `--image_dir path/to.png`
* 其它扩散与 DUSt3R 参数保持不动（减少变量）

### 5.2 输出

在 `--out_dir/exp_name/` 下新增：

* `traj_auto.txt`（三行序列，满足 ViewCrafter 原生格式约束）([GitHub][1])
* `viz_traj.mp4`（复用 `generate_traj_txt(..., viz_traj=True)`）([GitHub][3])
* 以及原本就会有的：`render0.mp4`, `diffusion0.mp4`, `pcd0.ply` 等([GitHub][2])

### 5.3 主流程（伪代码）

位置：`ViewCrafter.nvs_single_view()` 的 mode 分支内（紧贴 `single_view_txt`）

```python
elif self.opts.mode == 'single_view_autotraj':
    # 1) 准备 anchor / pcd / imgs / intrinsics，与 single_view_txt 同源
    #    radius / elevation / world_point_to_obj 等完全复用现有流程 :contentReference[oaicite:20]{index=20}

    # 2) 自动规划：输出 keyframe 序列（长度 K，2<=K<=25，首元素为0）
    phi_seq, theta_seq, r_seq = plan_traj_sequences(
        c2ws_anchor=c2ws[-1:],   # 与 single_view_txt 一致使用最后一帧为 anchor :contentReference[oaicite:21]{index=21}
        pcd=pcd[-1], imgs=imgs[-1], masks=masks,
        H=H, W=W, focals=focals[-1:], principal_points=principal_points[-1:],
        opts=self.opts,
        viewcrafter=self,        # 用于调用 self.run_render(..., nbv=True)
    )

    # 3) 落盘（可选但强烈建议，便于复现）
    write_traj_txt(os.path.join(self.opts.save_dir, 'traj_auto.txt'),
                   phi_seq, theta_seq, r_seq)

    # 4) 复用原生轨迹生成与可视化
    camera_traj, num_views = generate_traj_txt(
        c2ws[-1:], H, W, focals[-1:], principal_points[-1:],
        phi_seq, theta_seq, r_seq,
        frame=self.opts.video_length,
        device=self.device,
        viz_traj=True,
        save_dir=self.opts.save_dir,
    )

    # 5) 后续渲染 + diffusion 与 single_view_txt 完全一致
```

---

## 6. `utils/auto_traj_planner.py` 设计（可直接照此实现）

### 6.1 Planner 参数（全部可从命令行注入）

在 `configs/infer_config.py` 新增：

* `--planner_keyframes`（int，默认 7；表示写入 txt 的 keyframe 数，而不是最终 25 帧）
* `--planner_phi_max`（float，默认 45；候选最大水平旋转范围，匹配 d_phi 约束）([GitHub][1])
* `--planner_theta_max`（float，默认 30；候选最大俯仰范围，匹配 d_theta 约束）([GitHub][1])
* `--planner_r_max`（float，默认 0.2；匹配 d_r 约束）([GitHub][1])
* `--planner_phi_step`（float，默认 5）
* `--planner_theta_step`（float，默认 5）
* `--planner_r_step`（float，默认 0.05）
* `--planner_score`（str：`min_visible`/`max_visible`，默认 `min_visible`）
* `--planner_smooth_lambda`（float，默认 0.1；对相邻 keyframe 差分加惩罚，保证平滑）
* `--planner_loop_back`（bool，默认 True；是否让最后一个 keyframe 回到 0，形成 loop，便于稳定首尾）

  * ViewCrafter 在 `single_view_txt` 下如果 `phi[-1]==theta[-1]==r[-1]==0`，会把最后一帧强制设为原图（可利用这个机制稳定闭环）。([GitHub][2])

### 6.2 核心函数签名

```python
def plan_traj_sequences(
    c2ws_anchor, pcd, imgs, masks,
    H, W, focals, principal_points,
    opts, viewcrafter
):
    """
    return:
      phi_seq   : List[float]  # d_phi keyframes, deg, starts with 0
      theta_seq : List[float]  # d_theta keyframes, deg, starts with 0
      r_seq     : List[float]  # d_r keyframes, unitless ratio, starts with 0
    """
```

### 6.3 候选采样（推荐实现：局部步进 + 轻量 beam）

因为 `traj_txt` 最终会被 `generate_traj_txt` 做 smooth 插值，keyframe 不宜太密；推荐：

* `K = planner_keyframes`（默认 7）
* 每一步只从当前位置附近采样一个“小邻域”候选集合（而不是全局网格爆炸）

候选集合（每步）：

* `dphi ∈ {0, ±phi_step, ±2*phi_step, ...}` 截断到 `phi_max`
* `dtheta ∈ {0, ±theta_step, ...}` 截断到 `theta_max`
* `dr ∈ {0, ±r_step, ...}` 截断到 `r_max`

并施加约束：

* 相邻 keyframe 的 `|Δdphi|, |Δdtheta|, |Δdr|` 不超过阈值（对应官方 tip：避免突变）。([GitHub][1])

### 6.4 候选评分（关键：只渲染 mask，不跑 diffusion）

对每个候选 keyframe `(dphi_k, dtheta_k, dr_k)`：

1. 构造一个“短轨迹”用于评估：

   * 最简单：只评估该 keyframe 作为目标视角（frame=1 或 frame=2）
   * 实现上你可以直接调用 `generate_traj_txt` 但传入长度为 2 的序列 `[prev, candidate]`，并令 `frame=2`，取最后一帧 mask
2. 调用：

   ```python
   _, viewmask = viewcrafter.run_render(..., nbv=True)
   ```
3. 计算 score：

   * `min_visible`：`score = viewmask.sum()`，选最小（偏“去遮挡/去空洞探索”）
   * `max_visible`：`score = -viewmask.sum()`，选最大（偏“重叠稳定/少漂移”）

再加平滑正则：

```python
score_total = score + smooth_lambda * (|Δdphi| + |Δdtheta| + α|Δdr|)
```

---

## 7. 与 ViewCrafter 现有函数的对齐点（实现时别偏）

### 7.1 Anchor 选择

`single_view_txt` 用的是 `c2ws = scene.get_im_poses().detach()[1:]`（最后一个 view 为 0 pose）并用 `c2ws[-1:]` 作为 anchor。([GitHub][2])
自动规划器也必须一致，否则轨迹语义会变。

### 7.2 `r` 的语义

`generate_traj_txt` 会执行：

```python
rs = rs * c2ws_anchor[0,2,3]
```

所以 planner 产出的 `dr` 只应在 `[-0.5, 0.5]` 这种比例范围内，而不是世界单位。([GitHub][3])

### 7.3 最终仍然走 `generate_traj_txt`

不要自己重写插值，直接产出 keyframe 序列，交给 ViewCrafter 原生插值（smooth/linear）来做，保证和官方行为一致、减少漂移变量。([GitHub][3])

---

## 8. 复现指南（你可以直接让 AI 工具照做）

### 8.1 Baseline（原生 txt）

```bash
python inference.py \
  --mode single_view_txt \
  --image_dir ./test/images/fruit.png \
  --traj_txt ./assets/loop1.txt
```

（`loop1.txt` 是官方示例，格式见文档定义）([GitHub][1])

### 8.2 新功能（自动轨迹）

```bash
python inference.py \
  --mode single_view_autotraj \
  --image_dir ./test/images/fruit.png \
  --planner_keyframes 7 \
  --planner_score min_visible \
  --planner_phi_max 45 --planner_theta_max 30 --planner_r_max 0.2 \
  --planner_phi_step 5 --planner_theta_step 5 --planner_r_step 0.05 \
  --planner_smooth_lambda 0.1 \
  --planner_loop_back True
```

期望输出目录包含：

* `traj_auto.txt`
* `viz_traj.mp4`
* `render0.mp4`
* `diffusion0.mp4`
* `pcd0.ply`




