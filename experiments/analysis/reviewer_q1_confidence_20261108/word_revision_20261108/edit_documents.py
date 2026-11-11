import csv
import json
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph
from lxml import etree

ROOT = Path("/home/alireza/PythonProjects/tb-diagnosis")
WORD = ROOT / "docs/paper/my-paper/word"
BACKUP = Path("/tmp/afb-q7-word-review/originals")
RESULTS = ROOT / "experiments/analysis/reviewer_q1_confidence_20261108"
changes = []


def find(doc, prefix):
    matches = [p for p in doc.paragraphs if p.text.startswith(prefix)]
    assert len(matches) == 1, (prefix, len(matches))
    return matches[0]


def yellow(run):
    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return run


def replace_text(paragraph, old, new):
    assert paragraph.text.count(old) == 1, (old, paragraph.text)
    for run in paragraph.runs:
        if old in run.text:
            before, after = run.text.split(old, 1)
            run.text = before
            inserted = deepcopy(run._r)
            for child in list(inserted):
                if (
                    child.tag
                    != "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr"
                ):
                    inserted.remove(child)
            run._r.addnext(inserted)
            from docx.text.run import Run

            replacement = Run(inserted, paragraph)
            replacement.text = new
            yellow(replacement)
            if after:
                tail = deepcopy(run._r)
                replacement._r.addnext(tail)
                Run(tail, paragraph).text = after
            changes.append({"old": old, "new": new})
            return
    raise AssertionError("Target spans runs; preserve fields and edit explicitly")


def paragraph_after(anchor, text, template, highlight=True, heading=False):
    el = OxmlElement("w:p")
    if template._p.pPr is not None:
        el.append(deepcopy(template._p.pPr))
    anchor._p.addnext(el)
    p = Paragraph(el, anchor._parent)
    run = p.add_run(text)
    if template.runs and template.runs[0]._r.rPr is not None:
        run._r.insert(0, deepcopy(template.runs[0]._r.rPr))
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(0, 0, 0)
    run.bold = bool(heading)
    # Existing template may carry highlights from earlier revisions.
    run.font.highlight_color = WD_COLOR_INDEX.YELLOW if highlight else None
    p.paragraph_format.keep_with_next = bool(heading)
    p.paragraph_format.keep_together = True
    changes.append({"added": text})
    return p


def replace_whole(paragraph, text):
    assert not paragraph._p.xpath(".//w:fldChar | .//m:oMath | .//w:drawing")
    props = (
        deepcopy(paragraph.runs[0]._r.rPr)
        if paragraph.runs and paragraph.runs[0]._r.rPr is not None
        else None
    )
    old = paragraph.text
    paragraph.clear()
    run = paragraph.add_run(text)
    if props is not None:
        run._r.insert(0, props)
    yellow(run)
    changes.append({"old": old, "new": text})


def save_document_xml_only(doc, source, destination):
    payload = etree.tostring(
        doc.element, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    temporary = destination.with_suffix(".q7.docx")
    with ZipFile(source) as original, ZipFile(temporary, "w") as updated:
        for info in original.infolist():
            updated.writestr(
                info,
                (
                    payload
                    if info.filename == "word/document.xml"
                    else original.read(info.filename)
                ),
            )
    temporary.replace(destination)


manuscript_path = WORD / "Manuscript 14050514.docx"
m = Document(BACKUP / manuscript_path.name)
normal = find(m, "The original training and evaluation experiments")
mc = find(m, "Monte Carlo Dropout was used to estimate")
replace_text(
    mc,
    "During inference, dropout remained active for 10 stochastic forward passes.",
    "For this separate uncertainty analysis, dropout remained active for 10 stochastic forward passes.",
)
yellow(
    mc.add_run(
        " The single-pass accuracy evaluation and bootstrap confidence intervals in Section 4.7 used dropout-disabled predictions; MC Dropout variability was not used to calculate those intervals."
    )
)

precision = find(m, "Precision, Recall, and F1.")
replace_text(precision, "exceeds 0.5", "is at least 0.5")
metric_intro = find(m, "To assess detection and segmentation performance,")
replace_whole(
    metric_intro,
    "Detection and segmentation were assessed using the metrics described below. Ultralytics validation metrics for YOLO predictions and the connected-component metrics for refined binary masks use different implementations and are distinguished explicitly.",
)
ap = find(m, "Mean Average Precision (mAP).")
replace_whole(
    ap,
    "Ultralytics Mean Average Precision (mAP). For YOLO predictions, Ultralytics computes confidence-ranked average precision from the precision–recall curve. Mask and bounding-box mAP@50 are evaluated at an IoU threshold of 0.50, while mAP@50-95 averages results over thresholds from 0.50 to 0.95 in steps of 0.05.",
)
ap.runs[0].bold = False
ap.runs[0].italic = False
custom = paragraph_after(
    ap,
    "Custom Component Scores. Refined binary masks were evaluated by labeling four-connected foreground components and greedily matching predictions to reference components. Precision, recall, and F1 were calculated from pooled true-positive, false-positive, and false-negative counts at IoU ≥ 0.50. The refiner scores historically labeled mAP@50 and mAP@50-95 in Tables 7 and 10 are precision-based component scores averaged across images, with the latter also averaged over the ten IoU thresholds. These scores are not standard confidence-ranked AP and should not be interpreted as directly equivalent to Ultralytics mAP. Section 4.7 reports their confidence intervals under the explicit label of custom component scores.",
    normal,
)
method = paragraph_after(
    custom,
    "Bootstrap Confidence Intervals. Uncertainty in full-pipeline test performance was estimated by resampling the 1,000 DDS3 test images with replacement to generate 5,000 samples, each containing 1,000 image selections. Each selected image retained its predictions and reference annotations. Pooled component precision, recall, and F1 and image-mean custom scores were recalculated for each sample. The 2.5th and 97.5th percentiles defined the two-sided 95% confidence limits. The bootstrap random seed was 20261108. Resampling was performed at the complete mosaic-image level rather than at the pixel, bacillus, or constituent-patch level.",
    normal,
)
replace_text(
    normal,
    "During inference, a confidence threshold of 0.5 was applied to YOLO predictions.",
    "For the full-pipeline accuracy results reproduced in Section 4.7, the YOLO confidence threshold was 0.25 and the U-Net probability threshold was 0.5. Single-pass inference was used with dropout disabled; MC Dropout was assessed separately.",
)

for prefix in [
    "Table 7: Ablation study results",
    "Table 10: Comparison with previous methods",
]:
    p = find(m, prefix)
    yellow(
        p.add_run(
            " The refined-mask values labeled mAP are custom component scores defined in Section 3.3 and are not directly equivalent to Ultralytics mAP."
        )
    )

conclusion_heading = find(m, "Conclusion")
cost_heading = find(m, "Computational Cost and Resource Suitability")
previous = Paragraph(conclusion_heading._p.getprevious(), conclusion_heading._parent)
h = paragraph_after(
    previous,
    "Confidence Intervals for Full Pipeline Performance",
    cost_heading,
    heading=True,
)
p = paragraph_after(
    h,
    "A supplementary evaluation quantified test-sample uncertainty for the saved YOLOv11l-seg and Attention U-Net checkpoints trained with spatial debiasing. No models were retrained. All 1,000 DDS3 test images and their filename-matched masks were evaluated using YOLO confidence 0.25, a U-Net probability threshold of 0.5, FP32 single-pass inference, and a U-Net batch size of 10. YOLO processed images at 416 × 416 pixels and U-Net at 400 × 400 pixels. The original test-image order and preprocessing were retained, and neither the source dataset nor the model checkpoints were changed. Evaluation used PyTorch 2.7.1+cu126, Ultralytics 8.3.227, and an NVIDIA GeForce RTX 4060 Ti GPU with 16 GB VRAM, separately from the historical software setup in Section 4.1.",
    normal,
)
p = paragraph_after(
    p,
    "The saved component precision, recall, and both custom component scores were reproduced exactly. F1 was calculated from the pooled component counts. Table 13 presents these estimates with 95% confidence intervals calculated using the bootstrap procedure in Section 3.3. These intervals concern the full pipeline only; no paired bootstrap comparison between ablation configurations was performed.",
    normal,
)
caption = paragraph_after(
    p,
    "Table 13: Full-pipeline performance on the 1,000-image DDS3 test set with two-sided 95% percentile confidence intervals from 5,000 image-level bootstrap samples. Component precision, recall, and F1 use pooled counts at IoU ≥ 0.50. Custom scores preserve the original refiner calculation and are not standard confidence-ranked mAP.",
    normal,
)
caption.paragraph_format.keep_with_next = True
intervals = {
    r["metric"]: r
    for r in csv.DictReader((RESULTS / "confidence_intervals.csv").open())
}
table = m.add_table(rows=1, cols=3)
table.style = m.tables[-2].style
table.autofit = False
widths = [2.9, 1.0, 2.6]
for column, width in zip(table.columns, widths):
    column.width = Inches(width)
for cell, label in zip(
    table.rows[0].cells, ["Metric", "Estimate", "95% confidence interval"]
):
    cell.text = label
for key, label in [
    ("precision", "Component precision"),
    ("recall", "Component recall"),
    ("f1", "Component F1"),
    ("legacy_ap50", "Custom score @50"),
    ("legacy_ap50_95", "Custom score @50–95"),
]:
    row = table.add_row()
    values = intervals[key]
    content = [
        label,
        f"{float(values['estimate']):.4f}",
        f"{float(values['ci95_lower']):.4f}–{float(values['ci95_upper']):.4f}",
    ]
    for cell, value in zip(row.cells, content):
        cell.text = value
for row_index, row in enumerate(table.rows):
    pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    pr.append(cant_split)
    if row_index == 0:
        pr.append(OxmlElement("w:tblHeader"))
    for cell, width in zip(row.cells, widths):
        cell.width = Inches(width)
        for paragraph in cell.paragraphs:
            paragraph.paragraph_format.space_after = Pt(4)
            paragraph.paragraph_format.space_before = Pt(4)
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.paragraph_format.keep_with_next = row_index < len(table.rows) - 1
            for run in paragraph.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(11)
                run.font.color.rgb = RGBColor(0, 0, 0)
                run.bold = row_index == 0
                yellow(run)
caption._p.addnext(table._tbl)
# Insert the closing paragraph after the new table.
el = OxmlElement("w:p")
table._tbl.addnext(el)
after_table = Paragraph(el, caption._parent)
closing = paragraph_after(
    after_table,
    "The confidence intervals quantify uncertainty associated with the available test images for these fixed checkpoints. They do not assess variability across independent training runs or establish the statistical significance of differences between ablation configurations. Patient and slide identifiers were unavailable, so potential dependence between images from the same patient or slide could not be accounted for. These results do not establish external generalizability or clinical reliability.",
    normal,
)
after_table._p.getparent().remove(after_table._p)

limitations = find(m, "Several limitations warrant acknowledgment.")
paragraph_after(
    limitations,
    "The bootstrap analysis provides image-level confidence intervals for fixed trained models, but variability across independent training runs was not assessed. Resampling could not account for unknown patient- or slide-level grouping. The intervals therefore describe test-sample uncertainty under the evaluated DDS3 protocol and do not establish broader reproducibility or clinical reliability.",
    normal,
)

save_document_xml_only(m, BACKUP / manuscript_path.name, manuscript_path)

responses_path = WORD / "Responses.docx"
r = Document(BACKUP / responses_path.name)
question = find(
    r, "Can confidence intervals or results from repeated experiments be provided"
)
answer = Paragraph(question._p.getnext(), question._parent)
assert answer.text.strip() == "Answer:"
# Remove the unused blank Heading 2 immediately after this unanswered entry.
following = answer._p.getnext()
if following is not None and not "".join(following.xpath(".//w:t/text()")).strip():
    following.getparent().remove(following)
template = r.paragraphs[4]
texts = [
    "Thank you for this comment. We calculated 95% bootstrap confidence intervals to quantify uncertainty in the reported full-pipeline test performance. Using the saved model checkpoints and predictions for the 1,000 DDS3 test images, we generated 5,000 samples by sampling complete images with replacement. Each sample contained 1,000 image selections. Metrics were recalculated for each sample, and the 2.5th and 97.5th percentiles defined the confidence limits. No models were retrained.",
    "The original component-level precision and recall were reproduced exactly. Precision was 0.9126 (95% CI: 0.9099–0.9151), recall was 0.9387 (95% CI: 0.9363–0.9409), and F1 was 0.9254 (95% CI: 0.9232–0.9276). The custom component score at IoU 0.50 was 0.9132 (95% CI: 0.9106–0.9158), and the custom score averaged over IoU thresholds 0.50–0.95 was 0.5213 (95% CI: 0.5193–0.5235). These custom scores preserve the original refiner calculation and are distinguished from standard confidence-ranked Ultralytics mAP.",
    "The reproduced accuracy evaluation used a YOLO confidence threshold of 0.25, a U-Net probability threshold of 0.5, and single-pass inference with dropout disabled. The existing Monte Carlo Dropout analysis provides complementary estimates of pixel-wise prediction variability and was evaluated separately. We have clarified this distinction and the evaluation settings in the manuscript.",
    "The bootstrap intervals quantify test-sample uncertainty for fixed trained models. They do not assess variability across independent training runs or establish significant differences between ablation configurations. Patient and slide identifiers were unavailable, so potential dependence between images from the same patient or slide could not be accounted for. We acknowledge these limitations and do not interpret these intervals as establishing external generalizability or clinical reliability.",
    "Manuscript changes:",
    "1. Methodology, U-Net Refinement, Architecture: Attention U-Net (Section 3.2.2, page 15): Clarified that the ten-pass Monte Carlo Dropout analysis was separate from the dropout-disabled single-pass accuracy evaluation and the bootstrap confidence intervals.",
    "2. Methodology, Evaluation Metrics (Section 3.3, pages 16–17): Clarified the inclusive IoU matching threshold, distinguished Ultralytics mAP from the refiner’s custom connected-component scores, and added the complete bootstrap method, resampling unit, sample count, confidence-limit calculation, and random seed.",
    "3. Experiments, Experimental Setup (Section 4.1, page 18): Corrected the description of the reproduced full-pipeline accuracy settings to detector confidence 0.25 and U-Net probability threshold 0.5, with dropout disabled. The separate computational benchmark in Section 4.6 retains its own recorded settings.",
    "4. Experiments, new Confidence Intervals for Full Pipeline Performance subsection (Section 4.7, pages 27–28), Table 13 (page 28): Added the frozen-checkpoint evaluation protocol, saved-score reproduction, estimates and 95% bootstrap confidence intervals, and limitations. Existing section and table numbers were retained. The captions of Tables 7 and 10 now point to the custom-score definition in Section 3.3.",
    "5. Conclusion, limitations (Section 5, page 29): Added the absence of independent training repetitions and the inability to account for unknown patient/slide grouping. All manuscript additions and replacements for this response are highlighted in yellow. Page locations refer to the rendered Word manuscript after this revision.",
]
cursor = answer
for text in texts:
    cursor = paragraph_after(cursor, text, template, highlight=False)
    cursor.runs[0].font.name = template.runs[0].font.name
    if text == "Manuscript changes:":
        cursor.runs[0].bold = True
        cursor.paragraph_format.keep_with_next = True
save_document_xml_only(r, BACKUP / responses_path.name, responses_path)
Path("/tmp/afb-q7-word-review/changes.json").write_text(json.dumps(changes, indent=2))
print(
    "Updated manuscript and Reviewer 6 Q7 response, preserving all non-document XML ZIP members."
)
