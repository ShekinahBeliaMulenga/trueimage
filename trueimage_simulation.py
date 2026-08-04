from manim import *
import numpy as np

# ══════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════
BG       = "#060912"
SURFACE  = "#0E1221"
SURFACE2 = "#161B30"
BORDER   = "#1E2D50"

C_BLUE   = "#38BDF8"   # primary
C_INDIGO = "#818CF8"   # secondary
C_PURPLE = "#C084FC"   # accent
C_GREEN  = "#34D399"   # success
C_RED    = "#F87171"   # danger
C_AMBER  = "#FBBF24"   # warning
C_ORANGE = "#FB923C"   # nudenet
C_SLATE  = "#94A3B8"   # muted text
C_WHITE  = "#F1F5F9"   # body text

SCENE_COLORS = [
    C_BLUE, C_INDIGO, C_PURPLE, C_ORANGE,
    C_GREEN, C_BLUE,  C_INDIGO, C_AMBER,
    C_PURPLE, C_GREEN,
]


# ══════════════════════════════════════════════════════
#  PRIMITIVE HELPERS
# ══════════════════════════════════════════════════════
def T(text, sz=24, col=C_WHITE, w=NORMAL, mono=False):
    return Text(text, font_size=sz, color=col, weight=w,
                font="Courier New" if mono else "")

def card(w, h, bc=BORDER, fc=SURFACE, cr=0.16):
    return RoundedRectangle(width=w, height=h, corner_radius=cr,
                            fill_color=fc, fill_opacity=1,
                            stroke_color=bc, stroke_width=1.6)

def rule(w=13, col=BORDER, op=0.6):
    return Line(LEFT*w/2, RIGHT*w/2, stroke_color=col,
                stroke_width=1.0, stroke_opacity=op)

def badge(text, col=C_BLUE, w=2.2, h=0.46):
    bg = RoundedRectangle(width=w, height=h, corner_radius=h/2,
                          fill_color=col, fill_opacity=0.15,
                          stroke_color=col, stroke_width=1.2)
    lb = T(text, sz=15, col=col, mono=True)
    lb.move_to(bg)
    return VGroup(bg, lb)

def icon_circle(symbol, col=C_BLUE, r=0.38):
    ring = Circle(radius=r, fill_color=col, fill_opacity=0.15,
                  stroke_color=col, stroke_width=2)
    sym  = T(symbol, sz=22, col=col)
    sym.move_to(ring)
    return VGroup(ring, sym)

def node(label, col=C_BLUE, w=3.0, h=0.72):
    bg = RoundedRectangle(width=w, height=h, corner_radius=0.14,
                          fill_color=SURFACE2, fill_opacity=1,
                          stroke_color=col, stroke_width=1.8)
    lb = T(label, sz=18, col=C_WHITE)
    lb.move_to(bg)
    return VGroup(bg, lb)

def connect(a, b, col=C_BLUE, sw=2.0):
    return Arrow(a.get_right(), b.get_left(), buff=0.08,
                 stroke_width=sw, color=col,
                 max_tip_length_to_length_ratio=0.22)

def connect_v(a, b, col=C_BLUE, sw=2.0):
    return Arrow(a.get_bottom(), b.get_top(), buff=0.06,
                 stroke_width=sw, color=col,
                 max_tip_length_to_length_ratio=0.22)


# ══════════════════════════════════════════════════════
#  SCENE CHROME  – persistent logo + divider
# ══════════════════════════════════════════════════════
def make_chrome(scene_obj):
    logo = T("TRUEIMAGE", sz=18, col=C_BLUE, w=BOLD)
    logo.to_corner(UL).shift(RIGHT*0.1 + DOWN*0.02)
    scene_obj.add(logo)
    return logo


# ══════════════════════════════════════════════════════
#  MAIN PRESENTATION
# ══════════════════════════════════════════════════════
class TrueImagePresentation(Scene):

    def construct(self):
        self.camera.background_color = BG
        self._s01_title()
        self._s02_problem()
        self._s03_architecture()
        self._s04_safety()
        self._s05_face_detection()
        self._s06_inference()
        self._s07_result()
        self._s08_retention()
        self._s09_stack()
        self._s10_outro()

    # ── fade all & add persistent chrome ─────────────
    def _transition(self, keep=None):
        targets = [m for m in self.mobjects if m is not keep]
        if targets:
            self.play(*[FadeOut(m) for m in targets], run_time=0.55)
        if keep:
            self.remove(keep)

    def _heading(self, number, title, col=C_BLUE):
        num  = T(f"{number:02d}", sz=11, col=col, mono=True)
        line = Line(ORIGIN, RIGHT*0.22, stroke_color=col, stroke_width=1.5)
        lbl  = T(title, sz=32, col=C_WHITE, w=BOLD)
        grp  = VGroup(num, line, lbl).arrange(RIGHT, buff=0.16)
        grp.to_corner(UL).shift(RIGHT*0.12 + DOWN*0.42)
        accent = Line(grp.get_left() + DOWN*0.28,
                      grp.get_right() + DOWN*0.28,
                      stroke_color=col, stroke_width=1.0, stroke_opacity=0.4)
        header = VGroup(grp, accent)
        self.play(FadeIn(grp, shift=DOWN*0.1), Create(accent), run_time=0.55)
        return header

    # ════════════════════════════════════
    #  01  TITLE
    # ════════════════════════════════════
    def _s01_title(self):
        # Wordmark
        word = T("TRUEIMAGE", sz=88, col=C_BLUE, w=BOLD)
        word.move_to(UP*0.9)

        sub = T("Deep Learning Detection of AI-Generated Human Face Images",
                sz=24, col=C_SLATE)
        sub.next_to(word, DOWN, buff=0.38)

        author = T("Shekinah B. Mulenga  ·  Copperbelt University",
                   sz=18, col=C_SLATE)
        author.next_to(sub, DOWN, buff=0.2)

        # Ruled lines flanking author
        rl = Line(LEFT*5.5, LEFT*0.14, stroke_color=C_BLUE,
                  stroke_width=1, stroke_opacity=0.4)
        rr = Line(RIGHT*0.14, RIGHT*5.5, stroke_color=C_BLUE,
                  stroke_width=1, stroke_opacity=0.4)
        rl.next_to(author, DOWN, buff=0.28)
        rr.next_to(author, DOWN, buff=0.28)

        # Tech badges row
        badges = VGroup(
            badge("Python 3.x"),
            badge("Flask 3.1",   C_INDIGO),
            badge("OpenCV 4.13", C_GREEN),
            badge("NudeNet",     C_ORANGE),
            badge("ONNX",        C_PURPLE),
        ).arrange(RIGHT, buff=0.3)
        badges.next_to(rl, DOWN, buff=0.32)

        self.play(FadeIn(word, shift=UP*0.5), run_time=0.9)
        self.play(FadeIn(sub, shift=UP*0.2),
                  FadeIn(author, shift=UP*0.15), run_time=0.7)
        self.play(Create(rl), Create(rr), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(b, scale=0.85) for b in badges],
                               lag_ratio=0.14), run_time=0.9)
        self.wait(1.8)
        self._transition()

    # ════════════════════════════════════
    #  02  PROBLEM STATEMENT
    # ════════════════════════════════════
    def _s02_problem(self):
        chrome = make_chrome(self)
        hdr = self._heading(2, "The Problem", C_INDIGO)

        # Split canvas: REAL  |  AI-GENERATED
        divv = Line(UP*2.8, DOWN*2.8, stroke_color=BORDER,
                    stroke_width=1.4).move_to(ORIGIN + DOWN*0.2)

        # LEFT — real face silhouette
        real_card = card(5.4, 4.6, bc=C_GREEN).move_to(LEFT*3.2 + DOWN*0.35)
        head_r = Circle(radius=0.5, fill_color=C_SLATE, fill_opacity=0.8, stroke_width=0)
        head_r.move_to(real_card.get_center() + UP*0.75)
        body_r = ArcBetweenPoints(real_card.get_center()+LEFT*1.1+DOWN*0.75,
                                   real_card.get_center()+RIGHT*1.1+DOWN*0.75,
                                   angle=-PI/2.3, color=C_SLATE, stroke_width=16)
        rlbl   = T("REAL",  sz=22, col=C_GREEN, w=BOLD)
        rlbl.move_to(real_card.get_bottom() + UP*0.45)

        # RIGHT — "glitchy" silhouette (offset eyes = AI tell)
        ai_card = card(5.4, 4.6, bc=C_RED).move_to(RIGHT*3.2 + DOWN*0.35)
        head_a = Circle(radius=0.5, fill_color=C_SLATE, fill_opacity=0.8, stroke_width=0)
        head_a.move_to(ai_card.get_center() + UP*0.75)
        eye_l  = Ellipse(width=0.3, height=0.14, fill_color=C_RED, fill_opacity=0.9, stroke_width=0)
        eye_r  = Ellipse(width=0.3, height=0.14, fill_color=C_RED, fill_opacity=0.9, stroke_width=0)
        eye_l.move_to(head_a.get_center() + LEFT*0.18 + UP*0.09)
        eye_r.move_to(head_a.get_center() + RIGHT*0.23 + UP*0.03)   # asymmetric
        body_a = ArcBetweenPoints(ai_card.get_center()+LEFT*1.1+DOWN*0.75,
                                   ai_card.get_center()+RIGHT*1.1+DOWN*0.75,
                                   angle=-PI/2.3, color=C_SLATE, stroke_width=16)
        ailbl  = T("AI-GENERATED", sz=22, col=C_RED, w=BOLD)
        ailbl.move_to(ai_card.get_bottom() + UP*0.45)

        # Question mark
        qmark = T("?", sz=52, col=C_AMBER, w=BOLD).move_to(UP*1.6)

        self.play(FadeIn(qmark, scale=0.5), run_time=0.5)
        self.play(Create(real_card), Create(ai_card), run_time=0.6)
        self.play(FadeIn(VGroup(head_r, body_r)), FadeIn(rlbl),
                  FadeIn(VGroup(head_a, body_a, eye_l, eye_r)), FadeIn(ailbl),
                  run_time=0.7)
        self.play(Create(divv), run_time=0.4)

        # Callout: asymmetry
        arr = Arrow(ai_card.get_center() + RIGHT*1.0 + UP*0.9,
                    eye_r.get_center() + RIGHT*0.22,
                    buff=0.06, color=C_AMBER, stroke_width=2.5)
        note = T("Spatial asymmetry\n— a key GAN artefact",
                 sz=17, col=C_AMBER)
        note.next_to(arr.get_start(), RIGHT, buff=0.14)

        self.play(GrowArrow(arr), FadeIn(note), run_time=0.6)
        self.wait(1.6)
        self._transition(keep=chrome)

    # ════════════════════════════════════
    #  03  SYSTEM ARCHITECTURE
    # ════════════════════════════════════
    def _s03_architecture(self):
        chrome = make_chrome(self)
        hdr = self._heading(3, "System Architecture", C_PURPLE)

        # ── Row 1: Browser  →  Flask App ──
        browser = node("Browser\n(User Upload)", C_BLUE, w=2.8, h=1.0)
        flask   = node("Flask 3.1\n:3000",       C_INDIGO, w=2.8, h=1.0)
        browser.move_to(LEFT*4.8 + UP*1.4)
        flask.move_to(  LEFT*1.6 + UP*1.4)
        arr_bf = connect(browser, flask, C_BLUE)

        # ── Row 2: three parallel processors ──
        nudenet = node("NudeNet\nSafety Check",    C_ORANGE, w=2.8, h=1.0)
        yunet   = node("YuNet ONNX\nFace Detect",  C_PURPLE, w=2.8, h=1.0)
        preproc = node("OpenCV\nPreprocess",        C_GREEN,  w=2.8, h=1.0)
        nudenet.move_to(LEFT*4.2 + DOWN*0.3)
        yunet.move_to(  ORIGIN   + DOWN*0.3)
        preproc.move_to(RIGHT*4.2+ DOWN*0.3)

        arr_fn = Arrow(flask.get_bottom(), nudenet.get_top(),
                       buff=0.06, color=C_ORANGE, stroke_width=2,
                       max_tip_length_to_length_ratio=0.22)
        arr_fy = Arrow(flask.get_bottom(), yunet.get_top(),
                       buff=0.06, color=C_PURPLE, stroke_width=2,
                       max_tip_length_to_length_ratio=0.22)
        arr_fp = Arrow(flask.get_bottom(), preproc.get_top(),
                       buff=0.06, color=C_GREEN, stroke_width=2,
                       max_tip_length_to_length_ratio=0.22)

        # ── Row 3: DL Model ──
        model = node("Deep Learning\nModel", C_INDIGO, w=3.2, h=1.0)
        model.move_to(ORIGIN + DOWN*1.95)

        arr_nm = Arrow(nudenet.get_bottom(), model.get_top(),
                       buff=0.06, color=C_INDIGO, stroke_width=2,
                       max_tip_length_to_length_ratio=0.22)
        arr_ym = Arrow(yunet.get_bottom(), model.get_top(),
                       buff=0.06, color=C_INDIGO, stroke_width=2,
                       max_tip_length_to_length_ratio=0.22)
        arr_pm = Arrow(preproc.get_bottom(), model.get_top(),
                       buff=0.06, color=C_INDIGO, stroke_width=2,
                       max_tip_length_to_length_ratio=0.22)

        # ── Row 4: Result → APScheduler ──
        result   = node("Result\nResponse",  C_BLUE,  w=2.6, h=1.0)
        sched    = node("APScheduler\n45 s purge", C_RED, w=2.6, h=1.0)
        result.move_to(LEFT*1.8  + DOWN*3.4)
        sched.move_to( RIGHT*1.8 + DOWN*3.4)

        arr_mr = Arrow(model.get_bottom(), result.get_top(),
                       buff=0.06, color=C_BLUE, stroke_width=2,
                       max_tip_length_to_length_ratio=0.22)
        arr_ms = Arrow(model.get_bottom(), sched.get_top(),
                       buff=0.06, color=C_RED, stroke_width=2,
                       max_tip_length_to_length_ratio=0.22)

        # Animate row by row
        self.play(FadeIn(browser), GrowArrow(arr_bf), FadeIn(flask), run_time=0.9)
        self.play(
            LaggedStart(
                AnimationGroup(GrowArrow(arr_fn), FadeIn(nudenet, scale=0.9)),
                AnimationGroup(GrowArrow(arr_fy), FadeIn(yunet,   scale=0.9)),
                AnimationGroup(GrowArrow(arr_fp), FadeIn(preproc, scale=0.9)),
                lag_ratio=0.2,
            ), run_time=1.1,
        )
        self.play(
            LaggedStart(GrowArrow(arr_nm), GrowArrow(arr_ym), GrowArrow(arr_pm), lag_ratio=0.1),
            FadeIn(model, scale=0.9),
            run_time=0.8,
        )
        self.play(
            AnimationGroup(GrowArrow(arr_mr), FadeIn(result, scale=0.9)),
            AnimationGroup(GrowArrow(arr_ms), FadeIn(sched,  scale=0.9)),
            run_time=0.7,
        )
        self.wait(2.0)
        self._transition(keep=chrome)

    # ════════════════════════════════════
    #  04  SAFETY SCREENING  (NudeNet)
    # ════════════════════════════════════
    def _s04_safety(self):
        chrome = make_chrome(self)
        hdr = self._heading(4, "Safety Screening  —  NudeNet 3.4.2", C_ORANGE)

        # Icon + description
        ic  = icon_circle("🛡", C_ORANGE, r=0.55)
        ic.move_to(LEFT*4.8 + UP*0.6)

        desc = T("Uploaded images pass through NudeNet before any\n"
                 "face detection or inference is attempted.",
                 sz=22, col=C_SLATE)
        desc.next_to(ic, RIGHT, buff=0.45)

        self.play(DrawBorderThenFill(ic), FadeIn(desc, shift=RIGHT*0.12), run_time=0.7)

        # Decision flow (vertical)
        inp   = node("Uploaded Image",       C_BLUE,   w=4.0)
        check = node("NudeNet ONNX Check",   C_ORANGE, w=4.0)
        safe  = node("✓  Safe — continue",   C_GREEN,  w=3.4)
        block = node("✗  Unsafe — rejected", C_RED,    w=3.4)

        inp.move_to(LEFT*1.0 + DOWN*0.85)
        check.move_to(LEFT*1.0 + DOWN*1.95)
        safe.move_to(LEFT*2.8 + DOWN*3.2)
        block.move_to(RIGHT*1.8+ DOWN*3.2)

        a1 = connect_v(inp, check, C_ORANGE)
        a2 = Arrow(check.get_bottom(), safe.get_top(),
                   buff=0.06, color=C_GREEN, stroke_width=2,
                   max_tip_length_to_length_ratio=0.22)
        a3 = Arrow(check.get_bottom(), block.get_top(),
                   buff=0.06, color=C_RED, stroke_width=2,
                   max_tip_length_to_length_ratio=0.22)
        lbl_y = T("safe",   sz=14, col=C_GREEN, mono=True).next_to(a2, LEFT,  buff=0.08)
        lbl_n = T("unsafe", sz=14, col=C_RED,   mono=True).next_to(a3, RIGHT, buff=0.08)

        self.play(FadeIn(inp), run_time=0.4)
        self.play(GrowArrow(a1), FadeIn(check), run_time=0.5)
        self.play(
            GrowArrow(a2), FadeIn(safe,  scale=0.9), FadeIn(lbl_y),
            GrowArrow(a3), FadeIn(block, scale=0.9), FadeIn(lbl_n),
            run_time=0.7,
        )
        self.wait(1.5)
        self._transition(keep=chrome)

    # ════════════════════════════════════
    #  05  FACE DETECTION  (YuNet ONNX)
    # ════════════════════════════════════
    def _s05_face_detection(self):
        chrome = make_chrome(self)
        hdr = self._heading(5, "Face Detection  —  YuNet 2023mar ONNX", C_PURPLE)

        # LEFT: face illustration
        frame = card(4.0, 4.8, bc=C_PURPLE).move_to(LEFT*3.3 + DOWN*0.3)

        head = Circle(radius=0.5, fill_color=C_SLATE, fill_opacity=0.85, stroke_width=0)
        head.move_to(frame.get_center() + UP*0.72)
        body = ArcBetweenPoints(frame.get_center()+LEFT*1.02+DOWN*0.85,
                                 frame.get_center()+RIGHT*1.02+DOWN*0.85,
                                 angle=-PI/2.2, color=C_SLATE, stroke_width=14)
        silhouette = VGroup(head, body)

        self.play(FadeIn(frame), FadeIn(silhouette), run_time=0.6)

        # Scan
        scan = Line(frame.get_left()+RIGHT*0.3, frame.get_right()+LEFT*0.3,
                    stroke_color=C_PURPLE, stroke_width=4)
        scan.move_to(frame.get_top() + DOWN*0.38)
        self.play(FadeIn(scan))
        self.play(scan.animate.move_to(frame.get_bottom()+UP*0.38),
                  run_time=1.3, rate_func=linear)
        self.play(FadeOut(scan))

        # Bounding box with corner brackets
        bbox = RoundedRectangle(width=1.55, height=2.05, corner_radius=0.07,
                                stroke_color=C_GREEN, stroke_width=2.8)
        bbox.move_to(frame.get_center() + UP*0.2)

        # Corner ticks
        ticks = VGroup()
        for dx, dy in [(-1,1),(1,1),(-1,-1),(1,-1)]:
            h = Line(ORIGIN, RIGHT*0.24*dx, stroke_color=C_GREEN, stroke_width=3)
            v = Line(ORIGIN, UP*0.24*dy,    stroke_color=C_GREEN, stroke_width=3)
            VGroup(h,v).move_to(bbox.get_corner([dx,dy,0]))
            ticks.add(VGroup(h,v))

        conf_chip = badge("conf = 0.97", C_GREEN, w=2.0, h=0.38)
        conf_chip.next_to(bbox, UP, buff=0.1)

        self.play(Create(bbox), run_time=0.5)
        self.play(LaggedStart(*[Create(tk) for tk in ticks], lag_ratio=0.08),
                  FadeIn(conf_chip), run_time=0.5)

        # RIGHT: model card
        mc = card(4.6, 4.8, bc=C_PURPLE).move_to(RIGHT*3.0 + DOWN*0.3)

        rows_data = [
            ("Model",     "YuNet 2023mar",    C_PURPLE),
            ("Backend",   "OpenCV 4.13 ONNX", C_WHITE),
            ("Input",     "320 × 320 px",     C_WHITE),
            ("Faces",     "1 detected",       C_GREEN),
            ("Conf",      "0.97",             C_GREEN),
            ("File",      "…yunet_2023mar.onnx", C_SLATE),
        ]
        rows_vg = VGroup()
        for key, val, vc in rows_data:
            k = T(key, sz=17, col=C_SLATE, mono=True)
            v = T(val, sz=17, col=vc)
            row = VGroup(k, v).arrange(RIGHT, buff=0.3)
            rows_vg.add(row)
        rows_vg.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        rows_vg.move_to(mc.get_center()).shift(LEFT*0.3)

        self.play(FadeIn(mc), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r, shift=UP*0.08) for r in rows_vg],
                               lag_ratio=0.1), run_time=0.9)
        self.wait(1.5)
        self._transition(keep=chrome)

    # ════════════════════════════════════
    #  06  AI INFERENCE
    # ════════════════════════════════════
    def _s06_inference(self):
        chrome = make_chrome(self)
        hdr = self._heading(6, "AI Inference", C_INDIGO)

        # Three-node pipeline
        n_in    = node("Cropped Face\n224 × 224", C_BLUE,   w=3.4, h=1.1)
        n_model = node("Deep Learning\nModel",    C_INDIGO, w=3.4, h=1.1)
        n_out   = node("Confidence\nScore",       C_PURPLE, w=3.4, h=1.1)
        VGroup(n_in, n_model, n_out).arrange(RIGHT, buff=1.1).move_to(UP*0.8)

        a1 = connect(n_in, n_model, C_INDIGO)
        a2 = connect(n_model, n_out, C_PURPLE)

        self.play(
            LaggedStart(FadeIn(n_in), GrowArrow(a1),
                        FadeIn(n_model), GrowArrow(a2),
                        FadeIn(n_out), lag_ratio=0.18),
            run_time=1.3,
        )
        self.play(Flash(n_model[0], color=C_INDIGO,
                        flash_radius=0.9, line_length=0.28), run_time=0.6)

        # Output JSON block
        code_bg = card(7.2, 1.5, bc=C_PURPLE, fc=SURFACE2, cr=0.12)
        code_bg.move_to(DOWN*0.85)
        code_txt = T('{\n  "label": "AI-GENERATED",\n  "score": 0.83\n}',
                     sz=20, col=C_PURPLE, mono=True)
        code_txt.move_to(code_bg)

        # Confidence bar
        bar_bg = RoundedRectangle(width=7.2, height=0.38, corner_radius=0.19,
                                   fill_color=SURFACE2, fill_opacity=1, stroke_width=0)
        bar_bg.move_to(DOWN*2.1)
        bar_fill = RoundedRectangle(width=7.2*0.83, height=0.38, corner_radius=0.19,
                                     fill_color=C_PURPLE, fill_opacity=0.9, stroke_width=0)
        bar_fill.move_to(bar_bg.get_left() + RIGHT*7.2*0.83/2)
        pct = T("83 %", sz=22, col=C_PURPLE, w=BOLD)
        pct.next_to(bar_bg, RIGHT, buff=0.22)
        bar_lbl = T("AI Probability", sz=17, col=C_SLATE)
        bar_lbl.next_to(bar_bg, LEFT, buff=0.22)

        self.play(FadeIn(code_bg), Write(code_txt), run_time=0.7)
        self.play(FadeIn(bar_bg), FadeIn(bar_lbl), run_time=0.35)
        self.play(GrowFromEdge(bar_fill, LEFT), FadeIn(pct), run_time=0.9)
        self.wait(1.5)
        self._transition(keep=chrome)

    # ════════════════════════════════════
    #  07  RESULT PAGE
    # ════════════════════════════════════
    def _s07_result(self):
        chrome = make_chrome(self)
        hdr = self._heading(7, "Analysis Report", C_BLUE)

        # Main card
        rc = card(10.2, 5.2, bc=C_BLUE, cr=0.2)
        rc.move_to(DOWN*0.35)

        # Verdict
        verdict = T("AI-GENERATED", sz=54, col=C_RED, w=BOLD)
        verdict.move_to(rc.get_center() + UP*1.6)

        # Score section
        score_lbl = T("Confidence Score", sz=20, col=C_SLATE)
        score_lbl.move_to(rc.get_center() + UP*0.62)

        track = RoundedRectangle(width=6.8, height=0.4, corner_radius=0.2,
                                  fill_color=SURFACE2, fill_opacity=1, stroke_width=0)
        track.next_to(score_lbl, DOWN, buff=0.18)
        fill = RoundedRectangle(width=6.8*0.83, height=0.4, corner_radius=0.2,
                                 fill_color=C_RED, fill_opacity=0.85, stroke_width=0)
        fill.move_to(track.get_left() + RIGHT*6.8*0.83/2)
        pct = T("83 %", sz=26, col=C_RED, w=BOLD).next_to(track, RIGHT, buff=0.22)

        # Divider
        dv = rule(8.4, BORDER, 0.8).move_to(rc.get_center() + DOWN*0.3)
        # Meta row
        meta_items = [
            ("Faces Found",    "1",               C_WHITE),
            ("Scan Time",      "2.41 s",          C_WHITE),
            ("Confidence",     "High",            C_GREEN),
            ("label", "AI-GENERATED",   C_RED),
        ]
        meta_vg = VGroup()
        for key, val, vc in meta_items:
            chip = card(2.2, 0.9, bc=BORDER)
            k = T(key, sz=13, col=C_SLATE).move_to(chip.get_center() + UP*0.17)
            v = T(val, sz=17, col=vc, w=BOLD).move_to(chip.get_center() + DOWN*0.16)
            meta_vg.add(VGroup(chip, k, v))
        meta_vg.arrange(RIGHT, buff=0.28).next_to(dv, DOWN, buff=0.32)

        self.play(FadeIn(rc), run_time=0.4)
        self.play(Write(verdict), run_time=0.7)
        self.play(FadeIn(score_lbl), FadeIn(track), run_time=0.35)
        self.play(GrowFromEdge(fill, LEFT), FadeIn(pct), run_time=0.9)
        self.play(Create(dv), run_time=0.35)
        self.play(LaggedStart(*[FadeIn(m, scale=0.92) for m in meta_vg],
                               lag_ratio=0.15), run_time=0.9)
        self.wait(1.8)
        self._transition(keep=chrome)

    # ════════════════════════════════════
    #  08  ZERO RETENTION
    # ════════════════════════════════════
    def _s08_retention(self):
        chrome = make_chrome(self)
        hdr = self._heading(8, "Zero Retention  —  scheduler.py", C_AMBER)

        desc = T("Flask-APScheduler runs every 15 s and hard-deletes\n"
                 "any processed_* file older than 45 seconds.",
                 sz=22, col=C_SLATE)
        desc.move_to(UP*1.55)
        self.play(FadeIn(desc, shift=UP*0.1), run_time=0.6)

        # Timeline bar
        tl = card(11.0, 0.9, bc=BORDER, fc=SURFACE2, cr=0.1)
        tl.move_to(UP*0.45)

        # Track fill representing 45 s
        tl_fill = RoundedRectangle(
            width=11.0*0.83, height=0.9, corner_radius=0.1,
            fill_color=C_AMBER, fill_opacity=0.12, stroke_width=0,
        ).move_to(tl.get_left() + RIGHT*11.0*0.83/2)

        # Marker dots + labels
        events = [
            (0.0,  "t = 0 s\nUpload",         C_BLUE),
            (0.33, "t = 15 s\nSweep #1",       C_SLATE),
            (0.66, "t = 30 s\nSweep #2",       C_SLATE),
            (0.83, "t = 45 s\nFile expires",   C_AMBER),
            (1.0,  "Purged ✓",                 C_GREEN),
        ]
        markers = VGroup()
        for frac, label_text, col in events:
            dot = Circle(radius=0.11, fill_color=col, fill_opacity=1, stroke_width=0)
            dot.move_to(tl.get_left() + RIGHT*(frac*10.6+0.2))
            lb  = T(label_text, sz=14, col=col, mono=True)
            lb.next_to(dot, UP, buff=0.16)
            markers.add(VGroup(dot, lb))

        self.play(FadeIn(tl), run_time=0.4)
        self.play(GrowFromEdge(tl_fill, LEFT), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(m) for m in markers], lag_ratio=0.2),
                  run_time=0.9)

        # Real code snippet
        code_bg = card(9.0, 2.2, bc=C_GREEN, fc=SURFACE2, cr=0.14)
        code_bg.move_to(DOWN*2.15)

        code_src = (
            "max_age_seconds = 45\n"
            "for filename in os.listdir(upload_dir):\n"
            "    if filename.startswith('processed_'):\n"
            "        if file_age > max_age_seconds:\n"
            "            os.remove(filepath)   # hard delete"
        )
        code_txt = T(code_src, sz=17, col=C_GREEN, mono=True)
        code_txt.move_to(code_bg)

        self.play(FadeIn(code_bg), run_time=0.4)
        self.play(Write(code_txt), run_time=1.0)
        self.wait(1.8)
        self._transition(keep=chrome)

    # ════════════════════════════════════
    #  09  TECH STACK
    # ════════════════════════════════════
    def _s09_stack(self):
        chrome = make_chrome(self)
        hdr = self._heading(9, "Tech Stack", C_BLUE)

        stack = [
            ("Backend",          "Python  ·  Flask 3.1.3",                C_BLUE),
            ("Frontend",         "HTML  ·  CSS  ·  JavaScript",            C_INDIGO),
            ("Safety Screening", "NudeNet 3.4.2  (ONNX Runtime)",          C_ORANGE),
            ("Face Detection",   "OpenCV 4.13  +  YuNet 2023mar ONNX",    C_PURPLE),
            ("Image Processing", "Pillow 12.2  ·  NumPy 2.4",             C_GREEN),
            ("Zero Retention",   "Flask-APScheduler 1.13.1",               C_AMBER),
        ]

        rows = VGroup()
        for layer, detail, col in stack:
            row_bg = RoundedRectangle(
                width=11.4, height=0.68, corner_radius=0.13,
                fill_color=SURFACE2, fill_opacity=1,
                stroke_color=col, stroke_width=1.4,
            )
            # Coloured left accent bar
            accent = RoundedRectangle(
                width=0.18, height=0.68, corner_radius=0.05,
                fill_color=col, fill_opacity=0.9, stroke_width=0,
            ).move_to(row_bg.get_left() + RIGHT*0.09)

            layer_t  = T(layer,  sz=18, col=col, w=BOLD)
            detail_t = T(detail, sz=18, col=C_WHITE)
            layer_t.move_to(row_bg.get_left()  + RIGHT*2.0)
            detail_t.move_to(row_bg.get_right() + LEFT*3.5)

            rows.add(VGroup(row_bg, accent, layer_t, detail_t))

        rows.arrange(DOWN, buff=0.22).move_to(DOWN*0.18)

        self.play(
            LaggedStart(*[FadeIn(r, shift=UP*0.08) for r in rows], lag_ratio=0.1),
            run_time=1.4,
        )
        self.wait(1.8)
        self._transition(keep=chrome)

    # ════════════════════════════════════
    #  10  OUTRO
    # ════════════════════════════════════
    def _s10_outro(self):
        self._transition()

        word = T("TRUEIMAGE", sz=82, col=C_BLUE, w=BOLD)
        word.move_to(UP*0.95)

        tag = T("Explainable  ·  Privacy-Conscious  ·  AI Image Authenticity",
                sz=24, col=C_SLATE)
        tag.next_to(word, DOWN, buff=0.34)

        auth = T("Shekinah B. Mulenga  ·  Copperbelt University",
                 sz=18, col=C_SLATE)
        auth.next_to(tag, DOWN, buff=0.2)

        rl = Line(LEFT*5.4, LEFT*0.14, stroke_color=C_BLUE,
                  stroke_width=1, stroke_opacity=0.4)
        rr = Line(RIGHT*0.14, RIGHT*5.4, stroke_color=C_BLUE,
                  stroke_width=1, stroke_opacity=0.4)
        rl.next_to(auth, DOWN, buff=0.3)
        rr.next_to(auth, DOWN, buff=0.3)

        # Four final stat chips
        chips = VGroup(
            *[VGroup(
                card(2.55, 0.95, bc=BORDER),
                T(v, sz=22, col=C_PURPLE, w=BOLD),
                T(k, sz=14, col=C_SLATE),
              ) for k, v in [
                  ("Flask + ONNX",   "Runtime"),
                  ("YuNet",          "Face Model"),
                  ("NudeNet",        "Safety Filter"),
                  ("45 s",           "Max File Age"),
              ]]
        )
        for chip in chips:
            chip[1].move_to(chip[0].get_center() + UP*0.12)
            chip[2].move_to(chip[0].get_center() + DOWN*0.21)
        chips.arrange(RIGHT, buff=0.28).next_to(rl, DOWN, buff=0.38)

        self.play(FadeIn(word, shift=UP*0.4), run_time=0.9)
        self.play(FadeIn(tag, shift=UP*0.2),
                  FadeIn(auth, shift=UP*0.15), run_time=0.7)
        self.play(Create(rl), Create(rr), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(c, scale=0.9) for c in chips],
                               lag_ratio=0.15), run_time=0.9)
        self.wait(2.5)