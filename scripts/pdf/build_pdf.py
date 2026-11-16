"""Build the plain-language credit-increase mechanism PDF (reportlab)."""
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

HERE = Path(__file__).parent
FIG = HERE / "fig"
OUT = Path(sys.argv[1])
F = json.loads((FIG / "facts.json").read_text())

pdfmetrics.registerFont(TTFont("UI", r"C:\Windows\Fonts\segoeui.ttf"))
pdfmetrics.registerFont(TTFont("UI-B", r"C:\Windows\Fonts\segoeuib.ttf"))
pdfmetrics.registerFont(TTFont("UI-I", r"C:\Windows\Fonts\segoeuii.ttf"))
pdfmetrics.registerFont(TTFont("UI-SB", r"C:\Windows\Fonts\seguisb.ttf"))
pdfmetrics.registerFontFamily("UI", normal="UI", bold="UI-B", italic="UI-I", boldItalic="UI-B")

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#898781")
LINE = colors.HexColor("#e1e0d9")
ACCENT = colors.HexColor("#2a78d6")
ACCENT_BG = colors.HexColor("#eef4fc")
WARN_BG = colors.HexColor("#fdf3e7")
WARN_LINE = colors.HexColor("#eda100")
SOFT = colors.HexColor("#f6f5f2")

st = {
    "title": ParagraphStyle("title", fontName="UI-SB", fontSize=24, leading=29, textColor=INK, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName="UI", fontSize=12.5, leading=17, textColor=INK2, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="UI-SB", fontSize=16, leading=21, textColor=INK, spaceBefore=6, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="UI-SB", fontSize=12, leading=16, textColor=INK, spaceBefore=8, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="UI", fontSize=10.2, leading=15, textColor=INK, spaceAfter=6),
    "small": ParagraphStyle("small", fontName="UI", fontSize=8.6, leading=12, textColor=INK2),
    "caption": ParagraphStyle("caption", fontName="UI", fontSize=8.6, leading=12, textColor=INK2, spaceBefore=2, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="UI", fontSize=9, leading=12.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="UI-SB", fontSize=9, leading=12.5, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="UI-SB", fontSize=8.6, leading=12, textColor=INK2),
    "step": ParagraphStyle("step", fontName="UI", fontSize=8.6, leading=11.5, textColor=INK, alignment=TA_CENTER),
    "arrow": ParagraphStyle("arrow", fontName="UI", fontSize=14, leading=16, textColor=MUTED, alignment=TA_CENTER),
    "bullet": ParagraphStyle("bullet", fontName="UI", fontSize=10.2, leading=15, textColor=INK, leftIndent=14,
                             bulletIndent=2, spaceAfter=3),
}
P = lambda t, s="body": Paragraph(t, st[s])
W = A4[0] - 2 * 20 * mm


def bullets(items):
    return [Paragraph(i, st["bullet"], bulletText="•") for i in items]


def box(flowables, bg=ACCENT_BG, edge=ACCENT, pad=9):
    t = Table([[flowables]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
                           ("LEFTPADDING", (0, 0), (-1, -1), pad + 2), ("RIGHTPADDING", (0, 0), (-1, -1), pad),
                           ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad)]))
    return t


def table(rows, widths, header=True, zebra=False, bold_first_col=False):
    data = []
    for i, r in enumerate(rows):
        style = "cellh" if header and i == 0 else "cell"
        data.append([c if not isinstance(c, str) else
                     P(c, "cellb" if (bold_first_col and j == 0 and not (header and i == 0)) else style)
                     for j, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    ts = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
          ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if header:
        ts += [("LINEBELOW", (0, 0), (-1, 0), 1, INK2)]
    if zebra:
        ts += [("BACKGROUND", (0, i), (-1, i), SOFT) for i in range(2 if header else 1, len(rows), 2)]
    t.setStyle(TableStyle(ts))
    return t


def img(name, width):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(FIG / name)).getSize()
    return Image(str(FIG / name), width=width, height=width * ih / iw)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("UI", 7.8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 11 * mm, "Credit line increase: how the mechanism works  ·  demonstrative scenario, not a lending policy")
    canvas.drawRightString(A4[0] - 20 * mm, 11 * mm, str(doc.page))
    canvas.restoreState()


c102, c2, c1825, c10 = F["c102"], F["c2"], F["c1825"], F["c10"]
bd = F["band_default"]
pct = lambda v: f"{v * 100:.0f}%"
money = lambda v: f"{v:,.0f}"

story = []

# ---------------- Page 1 ----------------
story += [
    P("How the credit line increase check works", "title"),
    P("A plain-language guide to the scoring mechanism built in the Credit Knowledge Graph project", "subtitle"),
    box([P("<b>In one minute</b>", "body"),
         P("Imagine a customer asks for a higher credit limit. The mechanism answers one question: "
           "<i>does this customer's own track record support an increase of this size?</i>", "body"),
         *bullets([
             "It first measures the request <b>against the customer</b>, not in plain money: how many months of their usual "
             "payment would it take to pay back, and how big is it next to their current limit?",
             "Two <b>safety checks</b> come first. A customer who is very likely to miss a payment, or is already two months "
             "behind, is stopped. A customer with too little history is not judged at all.",
             "Everyone else is scored on <b>six factors</b>. Each factor earns between zero and full points, and the points add "
             "up to a <b>score from 0 to 100</b>.",
             "The score becomes a label (<b>Supportable</b>, <b>Review</b> or <b>Not supportable</b>), and a list of "
             "reasons shows exactly how many points each factor added or took away.",
         ]),
         ]),
    Spacer(1, 10),
    box([P("<b>Important:</b> this is a <b>demonstration</b>, not a bank's lending rule. The data (30,000 credit card "
           "customers in Taiwan, April–September 2005) contains no real requests for a higher limit and no record of what "
           "happened after one. The requested amount is something <i>we</i> choose to test, and the score shows how a "
           "transparent decision could be reasoned. It is not a decision.", "body")], bg=WARN_BG, edge=WARN_LINE),
    Spacer(1, 6),
    P("1. What the mechanism looks at", "h1"),
    P("For each customer we have six months of card history. From it the mechanism uses four kinds of information:"),
    table([
        ["What", "In everyday words", "Example"],
        ["Usage", "How much of the credit limit is being used (the bill divided by the limit).",
         "A 20,000 bill on a 100,000 limit is 20% usage."],
        ["Paying the bill", "What share of last month's bill the customer actually paid this month.",
         "Bill 10,000, then paid 5,000: half the bill paid."],
        ["Being late", "In how many of the six months the customer was behind on payments, and whether the latest "
         "month was one of them.", "Late in 2 of 6 months, latest month on time."],
        ["Predicted risk", "One number from our prediction model: the chance this customer misses next month's "
         "payment, learned from the history of all 30,000 customers.", "0.11 means about 11 in 100 customers like this one "
         "missed a payment."],
    ], [26 * mm, 84 * mm, W - 110 * mm], bold_first_col=True),
    Spacer(1, 6),
    P("Age, sex, education and marital status are <b>never used</b>. Payments are what the customer paid toward the card. "
      "They are <b>not income</b>; the data has no income at all.", "small"),
]

# ---------------- The five steps ----------------
story += [PageBreak(), P("2. The five steps at a glance", "h1"),
          P("Every request goes through the same five steps, in this order:")]
steps = [
    ("<b>1. Size the request</b><br/>in the customer's own units"),
    ("<b>2. Safety checks</b><br/>stop high risk, skip too little history"),
    ("<b>3. Score six factors</b><br/>each earns 0 to full points"),
    ("<b>4. Add up</b><br/>start at 50, add or subtract"),
    ("<b>5. Label + reasons</b><br/>Supportable / Review / Not"),
]
row, widths = [], []
bw, aw = (W - 4 * 7 * mm) / 5, 7 * mm
for i, s in enumerate(steps):
    row.append(P(s, "step")); widths.append(bw)
    if i < len(steps) - 1:
        row.append(P("→", "arrow")); widths.append(aw)
flow = Table([row], colWidths=widths, rowHeights=[22 * mm])
flow.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")] +
                         [("BACKGROUND", (i, 0), (i, 0), ACCENT_BG) for i in range(0, len(row), 2)] +
                         [("BOX", (i, 0), (i, 0), 0.8, ACCENT) for i in range(0, len(row), 2)]))
story += [flow, Spacer(1, 12)]

story += [
    P("Step 1: Measure the request against the customer, not in plain money", "h2"),
    P("The same amount means very different things for different people. Asking for 30,000 extra is a big step for "
      "someone who usually pays 1,000 a month (30 months' worth of payments), but a small one for someone who usually "
      "pays 20,000 a month (one and a half months). In the data, 30,000 is about 2.6 months of payment for customers "
      "among the biggest payers (top 5%) and about 92 months for customers among the smallest (bottom 5%)."),
    P("So every request is turned into two yardsticks taken from the customer's own history:"),
    *bullets([
        "<b>Payback months</b> = request ÷ usual monthly payment. The \"usual\" payment is the <i>middle</i> value of the "
        "six monthly payments (the median), so one unusually large payment does not distort it.",
        "<b>Request vs limit</b> = request ÷ current credit limit. Doubling a limit is a bigger step than adding 10% to it.",
    ]),
    P("The request itself can be set in three ways: a fixed amount for everyone (e.g. 30,000), a multiple of the "
      "customer's usual payment (e.g. 3× their usual payment), or a share of their current limit (e.g. 25% of the "
      "limit). The last two are \"own-scale\" requests, which treat every customer fairly no matter how big their account is."),
    P("Step 2: Two safety checks come before any scoring", "h2"),
    P("Some situations should not be decided by adding up points. The checks run first and <b>overrule the score</b>:"),
    table([
        ["Check", "It triggers when…", "Result"],
        ["Too risky", "The predicted chance of missing next month's payment is 50% or more, <b>or</b> the customer is "
         "two or more months behind in the latest month.", "<b>Not supportable</b>, whatever the score"],
        ["Too little history", "The card was hardly used (usage at or below 2% of the limit in all six months), "
         "<b>or</b> the customer paid nothing in at least four of the six months, so there is no usual payment to measure the "
         "request against.", "<b>Insufficient behavioural evidence</b>: the mechanism declines to judge"],
    ], [30 * mm, 95 * mm, W - 125 * mm], bold_first_col=True),
    Spacer(1, 6),
    P(f"Two real customers show why the checks matter. Customer 1825 would have scored <b>{c1825['score']:.0f}</b>, "
      f"which is above the Supportable line, but was two months behind in the latest month, so the result is "
      f"<b>Not supportable</b>. Customer 10 would have scored <b>{c10['score']:.0f}</b>, but their usual monthly "
      "payment is zero, so there is nothing to measure the request against. The result is <b>Insufficient evidence</b>, "
      "not a guess. If both checks trigger, \"Too risky\" wins."),
]

# ---------------- Six factors ----------------
story += [
    PageBreak(), P("Step 3: Score six factors", "h1"),
    P("Each factor asks one plain question. The answer earns somewhere between <b>zero</b> and <b>full</b> points for that "
      "factor. The <b>weight</b> says how much the factor matters in the final score; the weights add up to 100%."),
    table([
        ["Factor", "The question it asks", "Weight", "Full points when…", "Zero points when…"],
        ["Predicted risk", "How likely is this customer to miss next month's payment?", "30%", "chance is 10% or less",
         "chance is 60% or more"],
        ["Late-payment record", "How many of the six months were late? Is the latest month late?", "20%",
         "never late", "late in all six months (points are halved if the latest month was late)"],
        ["Payback months", "How many months of usual payment would the request equal?", "20%", "1 month or less",
         "30 months or more"],
        ["Share of bill paid", "In a typical month, what share of last month's bill did they pay?", "15%",
         "the whole bill", "nothing"],
        ["Request vs limit", "How big is the request next to the current limit?", "10%", "25% of the limit or less",
         "double the limit or more"],
        ["Current usage", "How much of the limit is in use right now?", "5%", "a moderate 10–50%",
         "never zero; lowest when maxed out (20% of points)"],
    ], [28 * mm, 52 * mm, 14 * mm, 35 * mm, W - 129 * mm], bold_first_col=True, zebra=True),
    Spacer(1, 8),
    P("The pictures below show exactly how each answer turns into points. Between the marked turning points the "
      "points change in a straight line, so there are no sudden jumps: a customer who pays 49% of the bill and one who pays "
      "51% get almost the same points."),
    img("factor_curves.png", W * 0.98),
    P("How each factor turns an answer into points. \"Full\" means the factor's whole weight; \"half\" means it is neutral.",
      "caption"),
]

story += [
    P("Where the turning points come from", "h2"),
    P("Most turning points were set by looking at what actually happened to customers in the data, using the training "
      "customers only:"),
    *bullets([
        "<b>Late months:</b> 11.5% of never-late customers missed the next payment, against about 70% of customers late in "
        "all six months. Recent lateness matters more than old lateness, which is why a late latest month halves the points.",
        "<b>Share of bill paid:</b> customers who paid nothing missed the next payment 58% of the time; those who paid "
        "half or more, about 15%.",
        "<b>Payback months:</b> where the request equalled a month or less of usual payment, 4.6% missed the next "
        "payment; where it equalled more than 30 months, 28.6% did.",
        "<b>Current usage</b> has a U-shape: unused cards (36%) and over-the-limit cards (35%) both missed payments far more "
        "often than cards used moderately (16%). So a moderate level earns the most, and \"never uses the card\" is "
        "<i>not</i> rewarded as safe.",
        "<b>Predicted risk:</b> the prediction model is adjusted (\"calibrated\") so that its number reads directly as an "
        "expected rate: 0.30 means roughly 30 in 100 similar customers missed a payment. Each customer's number comes "
        "from a model that never saw that customer's own outcome.",
        "<b>Request vs limit</b> is the one exception: the data says nothing about it, so its turning points are a stated "
        "<b>policy assumption</b> (large jumps relative to the limit are treated with caution) and it carries only 10%.",
    ]),
]

# ---------------- How the weights were chosen ----------------
P9 = json.loads((HERE.parents[1] / "outputs" / "reports" / "phase9_results.json").read_text(encoding="utf-8"))
stab = P9["stability"]
free = {r["band"]: r["miss_rate"] for r in P9["bands"]
        if r["scenario"] == "3 x typical payment" and r["version"] == "model removed completely"}
story += [
    KeepTogether([
        P("How the weights were chosen", "h2"),
        P("The turning points above came from the data. The <b>weights</b> (how much each factor counts) did not: they were "
          "set by judgement. That is a deliberate choice, not a shortcut."),
    ]),
    P("<b>Why the data could not choose them.</b> To learn weights from data you need a known right answer for each "
      "customer, here: \"was raising this customer's limit a good idea?\" The data has no limit increases and no record of "
      "what happened after one, so that answer does not exist. The only answer the data has is \"did the customer miss next "
      "month's payment?\" Learning weights from that would simply rebuild the prediction model, which is already the first "
      "factor, and would ignore the question of how big the request is."),
    P("<b>What the data did decide.</b> Besides the turning points, the data set the <i>order of importance</i>: factors "
      "with a stronger, clearer link to missed payments got more weight."),
    table([
        ["Factor", "Weight", "Why this size"],
        ["Predicted risk", "30%", "The single best summary of risk, and already a real rate. Largest weight, but not a "
         "majority, so behaviour and request size can still change the result."],
        ["Late-payment record", "20%", "Strongest single behaviour signal: 11.5% missed a payment with no late months, about "
         "70% with six."],
        ["Payback months", "20%", "The heart of the scenario: the request measured against what the customer has shown they "
         "can pay. Equal to the late-payment record, so request size matters as much as the strongest behaviour signal."],
        ["Share of bill paid", "15%", "Strong signal (58% vs about 15%), but it overlaps with predicted risk and late months."],
        ["Request vs limit", "10%", "Turning points are an assumption, not from data, so it is kept small."],
        ["Current usage", "5%", "U-shaped and noisy; unused cards are already caught by the safety check."],
    ], [32 * mm, 16 * mm, W - 48 * mm], bold_first_col=True),
    Spacer(1, 6),
    P("Some overlap is accepted on purpose: predicted risk already contains the late-payment history. It is kept so each "
      "reason can be shown on its own, and it is stated openly.", "small"),
    Spacer(1, 4),
    P("<b>How we checked that the choice does not matter too much.</b> Since the weights are judgement calls, the important "
      "question is whether a different reasonable choice would give different answers:"),
    *bullets([
        f"<b>Changing the weights:</b> each weight was raised and lowered by 20%, and each turning point moved by 10%, one at a "
        f"time ({stab['changes_tested']} changes). The order of customers barely moved (rank correlation at least "
        f"{stab['min_spearman']:.3f}, where 1 means identical), and at most {stab['max_band_flips'] * 100:.1f}% of customers "
        f"changed label.",
        f"<b>Removing the biggest factor entirely:</b> with predicted risk taken out, including its safety check, the labels "
        f"still sort real outcomes in the right order on the held-out customers: "
        f"{free['Supportable in this scenario'] * 100:.0f}% of Supportable, {free['Review'] * 100:.0f}% of Review and "
        f"{free['Not supportable in this scenario'] * 100:.0f}% of Not supportable customers missed the next payment "
        f"(request: 3 × usual payment). So the other five factors carry real information on their own.",
    ]),
    P("This shows the weights are <b>reasonable and stable</b>, not that they are <b>correct</b>: without real "
      "credit-increase outcomes, no one can show which weights would lead to the best lending decisions.", "small"),
]

# ---------------- Adding up ----------------
story += [
    Spacer(1, 6), P("Step 4: Add up the points", "h1"),
    P("Every customer starts at a neutral <b>50</b>. Each factor then pushes the score up or down:"),
    box([P("points for a factor = weight × (share of points earned − ½) × 100", "body")], bg=SOFT, edge=INK2),
    Spacer(1, 6),
    P("A factor that earns full points adds half its weight; one that earns nothing takes away half its weight; one that "
      "earns exactly half changes nothing. So the score can never go below 0 or above 100:"),
    table([
        ["Factor", "Weight", "Most it can add", "Most it can take away"],
        ["Predicted risk", "30%", "+15", "−15"],
        ["Late-payment record", "20%", "+10", "−10"],
        ["Payback months", "20%", "+10", "−10"],
        ["Share of bill paid", "15%", "+7.5", "−7.5"],
        ["Request vs limit", "10%", "+5", "−5"],
        ["Current usage", "5%", "+2.5", "−2.5"],
        ["<b>Total</b>", "<b>100%</b>", "<b>50 + 50 = 100</b>", "<b>50 − 50 = 0</b>"],
    ], [55 * mm, 25 * mm, 45 * mm, W - 125 * mm], bold_first_col=True),
    Spacer(1, 6),
    P("Starting from 50 is what makes the explanation readable: each reason can be stated as \"this added 7 points\" or "
      "\"this took away 5 points\", and those numbers always add up exactly to the final score (this is checked by an "
      "automated test)."),
    KeepTogether([P("Step 5: Turn the score into a label", "h2"), table([
        ["Score", "Label", "Meaning"],
        ["70 or more", "<b>Supportable in this scenario</b>", "The customer's record supports a request of this size."],
        ["50 to just under 70", "<b>Review</b>", "Mixed picture; a person should look at it."],
        ["Below 50", "<b>Not supportable in this scenario</b>", "The record does not support a request of this size."],
        ["(safety check)", "<b>Not supportable</b> / <b>Insufficient evidence</b>", "Set by step 2, whatever the score."],
    ], [35 * mm, 55 * mm, W - 90 * mm])]),
    Spacer(1, 6),
    P("The label always says <i>\"in this scenario\"</i> because it is tied to the amount asked for. The same customer can be "
      "Supportable for a small request and Review or Not supportable for a much larger one."),
]

# ---------------- Worked examples ----------------
def facts_table(c, extra):
    rows = [["Current limit", money(c["limit"])],
            ["Usual monthly payment", money(c["pay_median"])],
            ["Request tested (3 × usual payment)", f"{money(c['request'])}  →  {c['payback']:.1f} payback months, "
                                                    f"{c['to_limit'] * 100:.1f}% of limit"],
            ["Predicted chance of missing next payment", pct(c["pd"])],
            ["Late months (of 6)", f"{c['late']}{extra}"],
            ["Typical share of last bill paid", pct(c["coverage"])],
            ["Usage in latest month", pct(c["util"])]]
    return table(rows, [70 * mm, W - 70 * mm], header=False, bold_first_col=True)


story += [
    Spacer(1, 10),
    KeepTogether([
        P("3. Two worked examples", "h1"),
        P("Both customers are real rows from the data, tested with the same own-scale request: <b>3 × their usual monthly "
          "payment</b>. The bars show how many points each factor added (blue) or took away (red) from the starting 50."),
        P(f"Customer 102: score {c102['score']:.0f} → Supportable", "h2"),
        facts_table(c102, ", latest month: card not used"),
        Spacer(1, 4),
        img("contrib_102.png", W * 0.86),
        P(f"50 + 14.2 + 10.0 + 7.0 + 5.0 + 2.0 − 5.3 = {c102['score']:.1f}, shown as {c102['score']:.0f}.", "caption"),
    ]),
    P("<b>In words:</b> a low predicted risk and a clean record earn most of the points. The request is small for this "
      "customer, about three months of their usual payment and 1% of their limit, so it earns most of those points too. "
      "The one weak spot is that this customer usually pays only a small part of the previous bill, which costs 5.3 "
      "points. The result is still well above 70."),
]
story += [
    KeepTogether([
        P(f"Customer 2: score {c2['score']:.0f} → Review", "h2"),
        facts_table(c2, " (April and August), latest month paid in full"),
        Spacer(1, 4),
        img("contrib_2.png", W * 0.86),
        P(f"50 + 7.0 + 5.0 + 3.3 + 2.1 + 1.3 + 0.2 = {c2['score']:.1f}, just under the Supportable line of 70.", "caption"),
    ]),
    P("<b>In words:</b> the request is just as modest as customer 102's, so it earns the same points on the two request "
      "factors (+7.0 and +5.0). The difference is the customer's record. A 35% predicted chance of missing a payment "
      "is close to neutral and earns almost nothing (+0.2, against +14.2 for customer 102), and two late months keep "
      "only two-thirds of the record points. Nothing is clearly bad, but nothing is strongly good either, so the case "
      "goes to <b>Review</b>."),
    box([P("<b>Where the knowledge graph comes in.</b> The printed explanation also pulls evidence from the Neo4j graph, "
           "which stores each customer's six months as a linked chain. For customer 2 it shows <i>which</i> months were "
           "late (2005-04 and 2005-08), that they recovered and then fell behind again (one \"relapse\"), and how often "
           "customers in the same situation were late the next month (19%). The graph does not change the score; it gives "
           "the reviewer the evidence behind it.", "body")]),
]

# ---------------- Why relative ----------------
pair = {(r["customer"], r["increase"]): r["score"] for r in F["pair"]}
amounts = [10000, 30000, 50000, 100000, 150000, 200000]
sh = list(F["shares"].values())
story += [
    PageBreak(), P("4. Why the request is measured against the customer", "h1"),
    P("<b>A matched pair.</b> Two real customers look almost identical: the same 160,000 limit, the same low predicted "
      "risk (about 6%), and the same paying and usage habits. The one difference is that customer A usually pays about "
      "20,000 a month and customer B about 10,000. Here is what happens as the request grows:"),
    table([["Request", *[money(a) for a in amounts]],
           ["Customer A (usual payment ≈ 20,000)", *[f"{pair[('A', a)]:.1f}" for a in amounts]],
           ["Customer B (usual payment ≈ 10,000)", *[f"{pair[('B', a)]:.1f}" for a in amounts]],
           ["Gap", *[f"{pair[('A', a)] - pair[('B', a)]:+.1f}" for a in amounts]]],
          [58 * mm] + [(W - 58 * mm) / 6] * 6, bold_first_col=True),
    Spacer(1, 6),
    P("For the same amount, B needs twice as many months of payments to cover it. At 10,000 both are within a month "
      "or so and get essentially full marks; from 30,000 upward B scores lower, and the gap opens up to about 5 points as "
      "the request grows. This is the mechanism doing what it should: the request is judged by what the customer has "
      "shown they can pay, not by its size in money."),
    P("<b>Across all 30,000 customers.</b> Split customers into three equal groups by credit limit and compare two "
      "ways of setting the request:"),
    img("fixed_vs_relative.png", W * 0.86),
    P("Share of each group labelled Supportable.", "caption"),
    P(f"With a fixed 30,000 for everyone, only {pct(sh[0][0])} of the low-limit group is labelled Supportable, mostly because "
      f"30,000 is simply a large amount for small accounts. When each customer asks for 3 × their own usual payment, "
      f"that rises to {pct(sh[1][0])}, close to the high-limit group's {pct(sh[1][2])}. The gap that remains reflects "
      "real differences in behaviour, not account size."),
]

# ---------------- Does it make sense ----------------
story += [
    PageBreak(), P("5. Do the labels line up with what actually happened?", "h1"),
    P(f"There is no record of real limit increases, so the score cannot be checked against them. What we <i>can</i> check "
      f"is whether customers given a worse label really did go on to miss more payments. The {F['n_test']:,} customers "
      "below were held back from all model building and used once for this check (request: 3 × usual payment)."),
    img("band_default.png", W * 0.86),
    P("Share of customers in each label who missed the next month's payment.", "caption"),
    P(f"Supportable customers missed the next payment {pct(bd['Supportable in this scenario'][0])} of the time, half the "
      f"average of {pct(F['base_test'])}. Review sits at {pct(bd['Review'][0])} and Not supportable at "
      f"{pct(bd['Not supportable in this scenario'][0])}. The labels sort customers in the right order. Since predicted risk "
      "is the largest factor, part of this is expected; the check confirms that the other factors and the safety checks "
      "do not scramble that order."),
    P("Other checks that were run", "h2"),
    *bullets([
        "<b>Direction tests:</b> on 3,000 real customers, asking for more, a higher predicted risk or more late months "
        "never raises the score, and paying more of the bill never lowers it.",
        "<b>Fairness to account size:</b> multiplying a customer's limit, payments and request by the same factor leaves "
        "the score unchanged.",
        "<b>Stability:</b> changing every weight by ±20% or every turning point by ±10% keeps the ranking almost "
        "identical and changes the label for at most 3% of customers.",
        "<b>Demographics:</b> changing age, sex, education or marital status never changes the score.",
        "<b>Explanations add up:</b> the points listed in every explanation sum exactly to the score.",
    ]),
    P("6. What this mechanism is not", "h1"),
    *bullets([
        "<b>Not a lending decision.</b> No real increase requests or their outcomes exist in the data; the mechanism shows how "
        "a transparent decision could be reasoned.",
        "<b>Not about income.</b> Payments show what the customer paid toward the card, not what they earn.",
        "<b>Not a forecast of the new limit's effect.</b> The \"usage after the increase\" figure is shown in the explanation "
        "but not scored, because it simply assumes the bill stays the same.",
        "<b>Not causal.</b> The prediction model finds patterns in past behaviour; it does not say that an increase would "
        "cause or prevent a missed payment.",
        "<b>Limited history.</b> Six months of one card per customer, from 2005. The weights are open design choices, kept "
        "in one settings file (config/scoring.yaml) so they can be questioned and changed.",
    ]),
]

# ---------------- Glossary ----------------
story += [
    Spacer(1, 4),
    KeepTogether([
        P("Glossary", "h1"),
        table([
            ["Term", "Meaning in this document"],
            ["Usual monthly payment", "The middle value (median) of the customer's six monthly payments."],
            ["Payback months", "Request ÷ usual monthly payment."],
            ["Usage", "Bill ÷ credit limit for a month (negative bills count as zero)."],
            ["Share of bill paid", "Payment in a month ÷ the previous month's bill, capped at 100%. In this data a payment "
             "settles the previous month's bill, which was confirmed by checking the numbers."],
            ["Late month", "A month in which the customer was behind on payments."],
            ["Predicted risk", "The calibrated chance of missing next month's payment, from a model that never saw that "
             "customer's own outcome."],
            ["Relapse", "Late, then back on track, then late again; found by following the month-to-month chain in the graph."],
            ["Safety check", "A rule applied before scoring that overrules the score."],
        ], [42 * mm, W - 42 * mm], bold_first_col=True),
    ]),
]

doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                      bottomMargin=20 * mm, title="How the credit line increase check works",
                      author="Credit Knowledge Graph project",
                      subject="Plain-language explanation of the credit-increase suitability mechanism")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=footer)])
doc.build(story)
print("wrote", OUT)
