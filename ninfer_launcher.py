# -*- coding: utf-8 -*-
"""NInfer 图形界面启动器：用鼠标点选参数搭配，实时预览命令并启动 serve。

零外部依赖（仅 Python 自带 tkinter）。双击 启动器.bat 即弹窗口。
"""
import json
import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, simpledialog

# ---------------------------------------------------------------
# 常量：路径与基础参数
# ---------------------------------------------------------------
BASE_DIR     = r"J:\Bonsai\landing\repos\ninfer-4090-windows"
BUILD_DIR    = os.path.join(BASE_DIR, "_build_5080")
APPS_DIR     = os.path.join(BUILD_DIR, "apps")
ARTIFACT     = r"J:\Bonsai\landing\artifacts\Ternary-Bonsai-2-27B.ninfer"
CONFIG_FILE  = os.path.join(r"J:\Bonsai", "ninfer_launcher_profiles.json")
PORT_DEFAULT = 18787

# ---------------------------------------------------------------
# 维度定义：key -> {value: (显示名, 附加参数列表)}
# ---------------------------------------------------------------
BUILD_OPTIONS = {
    "new": ("新版(A+B, apps)", os.path.join(APPS_DIR, "ninfer-serve.exe")),
    "old": ("旧版(09-22, 根)", os.path.join(BUILD_DIR, "ninfer-serve.exe")),
}
KV_OPTIONS = {
    "fp8":  ("fp8",  ["--kv-dtype", "fp8"]),
    "bf16": ("bf16", ["--kv-dtype", "bf16"]),
    "k8v4": ("k8v4", ["--kv-dtype", "k8v4"]),
}
CTX_OPTIONS = {
    "32k":  ("32K",   ["--max-context", "32768"]),
    "128k": ("128K",  ["--max-context", "131072"]),
    "224k": ("224K",  ["--max-context", "224000"]),
    "256k": ("256K",  ["--max-context", "262144"]),
}
# 投机解码：键 -> (显示名, 参数)。k0=无, k1..k5=MTP, d1..d15=DFlash2
SPEC_OPTIONS = {"k0": ("无", [])}
SPEC_OPTIONS.update({f"k{i}": (f"MTP K={i}", ["--spec", "mtp", "--draft-tokens", str(i)])
                     for i in range(1, 6)})
SPEC_OPTIONS.update({f"d{i}": (f"DFlash2 K={i}",
                               ["--spec", "dflash2", "--draft-tokens", str(i), "--lm-head-draft"])
                     for i in range(1, 16)})
TB_OPTIONS = {
    "none":  ("无(不限)", []),
    "16000": ("16000",   ["--default-thinking-budget", "16000"]),
    "24000": ("24000",   ["--default-thinking-budget", "24000"]),
}

DIMENSIONS = [
    ("build", "构建版本", BUILD_OPTIONS),
    ("kv",    "KV类型",   KV_OPTIONS),
    ("ctx",   "上下文",   CTX_OPTIONS),
    ("spec",  "投机解码", SPEC_OPTIONS),
    ("tb",    "思考预算", TB_OPTIONS),
]

# 默认选中值
DEFAULTS = {"build": "new", "kv": "fp8", "ctx": "32k", "spec": "k2", "tb": "none"}


# ---------------------------------------------------------------
# 校验
# ---------------------------------------------------------------
def validate(combo):
    kv = combo.get("kv"); ctx = combo.get("ctx"); spec = combo.get("spec")
    if ctx == "256k" and kv != "k8v4":
        return "256K 上下文只支持 k8v4（fp8/bf16 显存不够）"
    if ctx == "224k" and kv != "k8v4":
        return "224K 上下文只支持 k8v4"
    if spec.startswith("d"):
        # 224K/256K 显存必须 k8v4（bf16 放不下）；DFlash 草案用独立 bf16 小窗口，不受主 KV 影响
        if ctx in ("224k", "256k"):
            if kv != "k8v4":
                return "DFlash2 + 长上下文只能用 k8v4 KV"
        elif ctx == "32k":
            if kv != "bf16":
                return "DFlash2 在 32K 建议用 bf16 KV（中文接受率场景）"
    return None


# ---------------------------------------------------------------
# 组合 -> 命令
# ---------------------------------------------------------------
def build_command(combo, port=None):
    port = port or PORT_DEFAULT
    build_val = combo.get("build", "new")
    exe = BUILD_OPTIONS[build_val][1] if build_val in BUILD_OPTIONS else BUILD_OPTIONS["new"][1]
    cmd = [exe, ARTIFACT, "--host", "127.0.0.1", "--port", str(port)]
    cmd.extend(KV_OPTIONS[combo.get("kv", "bf16")][1])
    cmd.extend(CTX_OPTIONS[combo.get("ctx", "32k")][1])
    cmd.extend(SPEC_OPTIONS[combo.get("spec", "k0")][1])
    tb = combo.get("tb", "none")
    if tb != "none":
        cmd.extend(TB_OPTIONS[tb][1])
    cmd.extend(["--temperature", "0.6", "--top-p", "0.95", "--tolerant-tool-calls"])
    return exe, cmd


# ---------------------------------------------------------------
# 命名组合（JSON 持久化）
# ---------------------------------------------------------------
def load_profiles():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_profiles(profiles):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------
# 图形界面
# ---------------------------------------------------------------
class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NInfer 启动器")
        self.root.geometry("880x680")
        self.root.minsize(760, 640)
        self._left_canvas = None  # 先声明，_build_widgets 里再建
        # 白底淡蓝
        self.bg = "#f4f8ff"
        self.fg = "#1a3a6b"
        self.accent = "#dcebff"
        self.root.configure(bg=self.bg)

        self.profiles = load_profiles()
        self.selection = dict(DEFAULTS)  # 当前选中值
        self._build_widgets()
        self._refresh()

    # ---- 控件 ----
    def _build_widgets(self):
        main = tk.Frame(self.root, bg=self.bg)
        main.pack(fill="both", expand=True, padx=14, pady=12)

        title = tk.Label(main, text="NInfer 启动器", font=("Microsoft YaHei UI", 15, "bold"),
                         bg=self.bg, fg="#1b4d8a")
        title.pack(anchor="w", pady=(0, 4))

        # 左：参数区（带垂直滚动条，保证所有参数组可滚动查看）
        left_outer = tk.Frame(main, bg=self.bg)
        left_outer.pack(side="left", fill="y", padx=(0, 14))
        self._left_canvas = tk.Canvas(left_outer, bg=self.bg, highlightthickness=0, width=180)
        self._left_scroll = ttk.Scrollbar(left_outer, orient="vertical",
                                          command=self._left_canvas.yview)
        self.left_inner = tk.Frame(self._left_canvas, bg=self.bg)
        self.left_inner.bind("<Configure>",
                             lambda e: self._left_canvas.configure(scrollregion=self._left_canvas.bbox("all")))
        self._left_canvas.create_window((0, 0), window=self.left_inner, anchor="nw")
        self._left_canvas.configure(yscrollcommand=self._left_scroll.set)
        self._left_canvas.pack(side="left", fill="y")
        self._left_scroll.pack(side="right", fill="y")
        # 滚轮支持
        self._left_canvas.bind("<Enter>", lambda e: self._left_canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self._left_canvas.bind("<Leave>", lambda e: self._left_canvas.unbind_all("<MouseWheel>"))

        self.var = {}   # 每个维度选中的 tk.StringVar
        self.radios = {}
        self.combos = {}   # 用下拉框的维度（spec）
        for key, label, opts in DIMENSIONS:
            box = tk.LabelFrame(self.left_inner, text=label, font=("Microsoft YaHei UI", 10),
                                bg=self.bg, fg="#2b3a6b")
            box.pack(fill="x", pady=4)
            self.var[key] = tk.StringVar(value=DEFAULTS[key])
            if key == "spec":
                # 投机解码：两级联动下拉 —— 类型（无/MTP/DFlash2）+ K 值
                f = tk.Frame(box, bg=self.bg)
                f.pack(anchor="w", padx=8, pady=4)
                self.var_spec_type = tk.StringVar(value="MTP")
                self.var_spec_k = tk.StringVar(value="2")
                self.combo_spec_type = ttk.Combobox(f, state="readonly", width=8,
                                                    textvariable=self.var_spec_type)
                self.combo_spec_type["values"] = ["无", "MTP", "DFlash2"]
                self.combo_spec_type.bind("<<ComboboxSelected>>", self._on_spec_change)
                self.combo_spec_type.pack(side="left", padx=(0, 6))
                self.combo_spec_k = ttk.Combobox(f, state="readonly", width=6,
                                                 textvariable=self.var_spec_k)
                self.combo_spec_k.bind("<<ComboboxSelected>>", self._on_spec_change)
                self.combo_spec_k.pack(side="left")
                self._update_spec_k_range()   # 设置 K 下拉范围 + 反映默认值
            else:
                self.radios[key] = []
                for val, (label_txt, _) in opts.items():
                    rb = tk.Radiobutton(box, text=label_txt, variable=self.var[key], value=val,
                                        command=self._on_change,
                                        bg=self.bg, fg="#2b3a6b", activebackground=self.bg,
                                        font=("Microsoft YaHei UI", 9))
                    rb.pack(anchor="w", padx=8)
                    self.radios[key].append(rb)

        # 右：命令预览 + 操作
        right = tk.Frame(main, bg=self.bg)
        right.pack(side="right", fill="both", expand=True)

        tk.Label(right, text="启动命令预览", bg=self.bg, fg="#9b8c5a",
                 font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w", pady=(0, 2))
        self.cmd_box = tk.Text(right, height=9, width=58, wrap="word",
                               bg="white", fg="#1a1a1a", bd=1, relief="solid",
                               font=("Consolas", 9))
        self.cmd_box.pack(fill="both", expand=True)
        self.cmd_box.configure(state="disabled")

        # 状态行
        self.status = tk.Label(right, text="", bg=self.bg, fg="#4a6b4a",
                               font=("Microsoft YaHei UI", 9))
        self.status.pack(anchor="w", pady=(4, 0))

        # 组合区
        combox = tk.Frame(right, bg=self.bg)
        combox.pack(fill="x", pady=(10, 4))

        tk.Label(combox, text="命名组合:", bg=self.bg, fg="#2b3a6b",
                 font=("Microsoft YaHei UI", 10)).pack(side="left")
        self.profile_cb = ttk.Combobox(combox, state="readonly", width=18,
                                       values=list(self.profiles.keys()))
        self.profile_cb.pack(side="left", padx=(6, 0))
        ttk.Button(combox, text="加载", command=self._load_profile).pack(side="left", padx=4)
        ttk.Button(combox, text="保存当前", command=self._save_profile).pack(side="left", padx=4)
        ttk.Button(combox, text="删除", command=self._del_profile).pack(side="left", padx=4)

        # 启动按钮
        ttk.Button(right, text="启 动", command=self._launch,
                   width=14).pack(pady=(8, 0))

    # ---- 事件 ----
    def _on_mousewheel(self, event):
        self._left_canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _on_change(self):
        for key, var in self.var.items():
            if key == "spec":
                self.selection[key] = self._spec_key_from_ui()
            else:
                self.selection[key] = var.get()
        self._refresh()

    def _update_spec_k_range(self):
        """按类型更新 K 下拉范围（MTP:1-5, DFlash2:1-15, 无:禁用）。"""
        t = self.var_spec_type.get()
        if t == "无":
            self.combo_spec_k["values"] = []
            self.combo_spec_k.set("")
            self.combo_spec_k.config(state="disabled")
        elif t == "MTP":
            self.combo_spec_k["values"] = [str(i) for i in range(1, 6)]   # 1-5
            self.combo_spec_k.config(state="readonly")
        else:  # DFlash2
            self.combo_spec_k["values"] = [str(i) for i in range(1, 16)]  # 1-15
            self.combo_spec_k.config(state="readonly")
        # 若当前 K 超出范围，重置为范围内的默认
        cur = self.var_spec_k.get()
        if cur not in self.combo_spec_k["values"]:
            self.var_spec_k.set(self.combo_spec_k["values"][0] if self.combo_spec_k["values"] else "")

    def _on_spec_change(self, event=None):
        """投机解码类型或 K 变化时联动。"""
        self._update_spec_k_range()
        self.selection["spec"] = self._spec_key_from_ui()
        self._refresh()

    def _spec_key_from_ui(self):
        """由类型+K 生成 spec 键（k0/k1..k5/d1..d15）。"""
        typ = self.var_spec_type.get()
        k = self.var_spec_k.get()
        if typ == "无":
            return "k0"
        if typ == "MTP":
            return f"k{k}" if k.isdigit() and 1 <= int(k) <= 5 else "k2"
        if typ == "DFlash2":
            return f"d{k}" if k.isdigit() and 1 <= int(k) <= 15 else "d7"
        return "k0"

    def _refresh(self):
        """刷新命令预览 + 校验提示"""
        err = validate(self.selection)
        _, cmd = build_command(self.selection)
        cmdline = " ".join(cmd)
        self.cmd_box.configure(state="normal")
        self.cmd_box.delete("1.0", "end")
        self.cmd_box.insert("1.0", cmdline)
        self.cmd_box.configure(state="disabled")
        if err:
            self.status.config(text="⚠ " + err, fg="#b03030")
        else:
            self.status.config(text="✓ 配置合法，可启动", fg="#4a6b4a")

    def _load_profile(self):
        name = self.profile_cb.get()
        if not name or name not in self.profiles:
            return
        combo = self.profiles[name]
        valid_keys = {d[0] for d in DIMENSIONS}
        valid_vals = {d[0]: set(d[2].keys()) for d in DIMENSIONS}
        for key, val in combo.items():
            if key == "spec":
                # spec 是组合键（k0/k1..k5/d1..d15），反向设置类型+K 两个下拉
                if val == "k0":
                    self.var_spec_type.set("无")
                elif val.startswith("k"):
                    self.var_spec_type.set("MTP")
                    self.var_spec_k.set(val[1:])
                elif val.startswith("d"):
                    self.var_spec_type.set("DFlash2")
                    self.var_spec_k.set(val[1:])
                self._update_spec_k_range()
            elif key in valid_keys and val in valid_vals[key]:
                self.var[key].set(val)
        self._on_change()

    def _save_profile(self):
        name = tk.simpledialog.askstring("保存组合", "输入组合名称：")
        if not name:
            return
        self.profiles[name] = dict(self.selection)
        save_profiles(self.profiles)
        self.profile_cb["values"] = list(self.profiles.keys())
        self.profile_cb.set(name)
        self.status.config(text=f"已保存组合 '{name}'", fg="#4a6b4a")

    def _del_profile(self):
        name = self.profile_cb.get()
        if not name or name not in self.profiles:
            return
        del self.profiles[name]
        save_profiles(self.profiles)
        self.profile_cb["values"] = list(self.profiles.keys())
        self.profile_cb.set("")

    def _launch(self):
        err = validate(self.selection)
        if err:
            messagebox.showerror("参数不合法", err)
            return
        exe, cmd = build_command(self.selection)
        # 确认框
        if not messagebox.askyesno("启动确认",
                                   f"确认启动 serve？\n\n{exe}\n\n参数:\n{' '.join(cmd)}"):
            return
        env = dict(os.environ)
        env["PATH"] = BUILD_DIR + os.pathsep + env.get("PATH", "")
        # 启动 serve：日志实时显示 + 同时落盘，关闭日志窗口即停 serve
        self._spawn_serve(cmd, env)

    def _spawn_serve(self, cmd, env):
        """启动 serve：真实 CMD 窗口实时显示日志 + 同时落盘到 logs\\serve_<时间戳>.log。
        通过 serve_tee.py 转发；关闭 CMD 窗口 = 停止 serve。
        """
        import time as _time
        ts = _time.strftime("%Y%m%d_%H%M%S")
        logfile = os.path.join(r"J:\Bonsai", "logs", f"serve_{ts}.log")
        tee = os.path.join(r"J:\Bonsai", "serve_tee.py")
        # 用 CREATE_NEW_CONSOLE 启动 serve_tee（真实 CMD 窗口），tee 负责显示+落盘
        subprocess.Popen([sys.executable, tee, logfile] + cmd,
                         env=env, cwd=BUILD_DIR,
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
        self.status.config(text="✓ 已启动（CMD 日志窗口，关闭即停）", fg="#4a6b4a")


def main():
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()