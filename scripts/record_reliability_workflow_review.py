"""Explicit assistant judgments on all64 fixed records, never expert labels."""
from collections import Counter
import json

from audit_reliability_workflows import OUT,INPUT,prepare
from dataset_catalog import read_rows,sha
from json_io import write_json

# Written after reading complete records including context, history and actions.
NOTES={
0:('Pass','The request to add a driver is paired with identity and driver checks before a policy update.'),
1:('Pass','The loss-run request is paired with compilation, accuracy verification and document delivery.'),
2:('Pass','The urgent roof concern is paired with a safety check and human emergency handoff; the shorter action list still covers those concerns.'),
3:('Unsure','The split-limit question and homeowners context need clarification; the workflow offers coverage options without explicitly resolving the policy context.'),
4:('Pass','The stated lawsuit threat is documented and sent to a human legal team rather than negotiated by the AI.'),
5:('Pass','The payment-plan request is matched with account lookup, plan choices and enrollment.'),
6:('Unsure','Condolences and policy verification fit the bereavement request, but collect_deceased does not clearly specify what information should be collected.'),
7:('Pass','An explicit supervisor request is acknowledged and routed to a human supervisor.'),
8:('Pass','The claim delay is addressed through lookup and checking for missing documents before explaining next steps.'),
9:('Pass','The quote discrepancy is investigated and escalated; having three actions instead of the source group’s four does not by itself show an incorrect workflow.'),
10:('Pass','The injury report and employer pushback are collected and sent to a human; precise rights-related response content is not independently verified.'),
11:('Unsure','A split-limit increase is requested with homeowners context; the workflow updates coverage without explicitly clarifying the relevant policy.'),
12:('Pass','The delayed claim is looked up and missing-document checks provide a relevant next step.'),
13:('Pass','The sold vehicle is removed after identity verification, with premium recalculation and updated documents.'),
14:('Pass','The history and current request contain a new claim plus an existing claim; the workflow handles both.'),
15:('Pass','Although context lists an active policy, the refund workflow verifies cancellation before checking refund status rather than assuming cancellation.'),
16:('Pass','The home-theft request is matched with policy verification, theft details, police-report guidance and an adjuster.'),
17:('Unsure','The record pairs an ID-card request with life-policy context; identity verification is present but the relevant document/policy is not clarified.'),
18:('Issue','The customer explicitly asks for a deductible, but the generated actions and response elements focus on breach response without an explicit deductible answer or lookup.'),
19:('Pass','The name-change request is paired with identity and documentation checks followed by an update.'),
20:('Unsure','The auto-accident request is paired with commercial-liability context and an ambiguous prior agent statement; verification is included but policy reconciliation is not explicit.'),
21:('Unsure','Password recovery fits the locked-out message, but the supplied app_error subcategory is less specific than the apparent account-access problem.'),
22:('Pass','The deductible question is paired with policy lookup and providing the deductible amount.'),
23:('Pass','The renewal increase is paired with policy lookup, rate-factor explanation and discount options.'),
24:('Unsure','The customer never identifies the uncovered incident; explaining an exclusion before explicitly clarifying the gap may be premature.'),
25:('Unsure','The delivery-driver question is ambiguous; the workflow lists coverage explanation but no explicit clarification of what happened.'),
26:('Pass','The adjuster-delay message is matched with claim lookup, status information and acknowledgment of frustration.'),
27:('Pass','The renewal-price question receives a rate-factor and discount workflow after policy lookup.'),
28:('Pass','The marriage-related name change is supported by identity/documentation checks and updated documents.'),
29:('Pass','The active-policy context conflicts with the requested cancellation letter, but verify_cancellation explicitly addresses that uncertainty before letter generation.'),
30:('Pass','The payment request is matched with account lookup, payment and confirmation.'),
31:('Pass','Policy lookup precedes the exclusion explanation and possible endorsement; actual coverage facts are not certified.'),
32:('Pass','The address change includes identity verification and an update, with a state matching the stated new address.'),
33:('Issue','The customer names a homeowners policy while context names commercial general liability; the workflow proceeds to an endorsement update without explicitly finding or reconciling the requested policy.'),
34:('Pass','The explicit bad-faith allegation is documented and handed to a human legal team.'),
35:('Unsure','The car valuation dispute conflicts with homeowners context; human handoff is appropriate but the policy mismatch is not explicitly resolved.'),
36:('Pass','The previously existing damage is flagged and assigned to an adjuster instead of automatically accepted.'),
37:('Pass','The beneficiary change is paired with identity verification and collection of the new beneficiary information.'),
38:('Unsure','The customer asks about auto/home gaps but context identifies life insurance; lookup and hybrid handling could help, but selecting the correct policies is not explicit.'),
39:('Pass','Stated job loss and inability to pay support hardship options and a human retention handoff.'),
40:('Unsure','The customer says they have flood insurance while context identifies renters insurance; the workflow does not explicitly select which policy will be explained.'),
41:('Unsure','The split-limit question is paired with workers-compensation context; policy clarification is missing from the listed steps.'),
42:('Issue','The conversation history says the payment has already been processed, but required_actions includes process_payment again before adding the vehicle.'),
43:('Pass','The open claim is looked up before a status update and expectations are provided.'),
44:('Pass','The wrong-car/life-policy discrepancy is documented and escalated for correction instead of automatically changing the policy.'),
45:('Pass','The delayed claim receives lookup and a status update without promising an unverified outcome.'),
46:('Pass','The declared denied-status scenario receives lookup, a denial explanation, appeal information and human handoff; real denial merits are not assessed.'),
47:('Pass','The spreading water-damage report is collected and verified, with mitigation and inspection steps.'),
48:('Unsure','The customer asks for a deductible after an ER visit, while the workflow emphasizes claim filing and general coverage; a specific deductible answer is not explicit.'),
49:('Pass','Lookup and providing status address the question, even though the source group normally also lists setting expectations.'),
50:('Pass','The distressed message receives empathy, needs assessment and human handoff; this is a workflow-coherence judgment, not a crisis-handling certification.'),
51:('Pass','Stated financial hardship is paired with account lookup, hardship options and human handoff.'),
52:('Pass','The explicit request for someone who understands seniors is paired with assistance and specialist handoff; three actions can still address the request.'),
53:('Pass','The claim-status request is addressed by lookup, a status update and expectations.'),
54:('Pass','The missing-upload report is investigated through upload/processing checks before requesting resubmission.'),
55:('Pass','The theft report is paired with details, policy verification and police-report guidance.'),
56:('Pass','The valuation dispute is documented, appraisal is offered and a human supervisor is involved.'),
57:('Unsure','Identity and document generation are included, but delivery appears only as a response element rather than an explicit required action.'),
58:('Pass','The certificate request is paired with identity verification, certificate compilation and delivery; the recipient question may need clarification in a real interaction.'),
59:('Unsure','The auto-accident request is paired with cyber-liability context; verification and handoff are present but identifying the relevant auto policy is not explicit.'),
60:('Pass','The duplicate-car-policy concern is investigated through review of both policies and human escalation rather than automatic cancellation.'),
61:('Pass','The claims-history request is paired with identity verification, history lookup and document presentation.'),
62:('Pass','The payment request is paired with account lookup, processing and confirmation.'),
63:('Issue','The flood question is paired with health-policy context, yet the workflow proceeds to a flood-exclusion explanation and recommendation without explicitly reconciling the policy.'),
}


def main():
    diagnostics=prepare();rows=read_rows(INPUT/'controlled_records.jsonl');assert set(NOTES)=={w['sample_id'] for w in rows}==set(range(64))
    judgments=[{'sample_id':w['sample_id'],'verdict':NOTES[w['sample_id']][0],'reason':NOTES[w['sample_id']][1],'reviewer':'unblinded_assistant','record':w['record']} for w in rows]
    (OUT/'ai_judgments.jsonl').write_text(''.join(json.dumps(w,ensure_ascii=False)+'\n' for w in judgments),encoding='utf-8')
    counts=dict(Counter(w['verdict'] for w in judgments));summary={'counts':counts,'n':64,'judgments_sha256':sha(OUT/'ai_judgments.jsonl'),
        'limits':'One unblinded AI reviewer, no expert/human calibration; no insurance/legal accuracy claim, no population estimate, no comparative semantic baseline score.'}
    write_json(OUT/'review_summary.json',summary)
    lines=['# Reliability workflow audit','',summary['limits'],'',
        'All64 prefix-cohort records were read, including context and conversation history. Source support is an objective diagnostic, not a correctness label. All judgments are available below.','',
        '## Why the action-count distribution drifted','',
        f'Training action counts: {diagnostics["source_action_counts"]}. Given the generated category/subcategory/difficulty mix, source examples imply expected counts {diagnostics["prefix"]["expected_counts_given_generated_groups"]}. Actual generated counts: {diagnostics["prefix"]["actual_action_counts"]}.',
        f'Nine records have action counts outside their observed source subgroup support. {diagnostics["prefix"]["records_with_unseen_actions"]}/64 contain at least one action label absent from training. Neither flag proves an incorrect workflow.','',
        'The shift is largely seven source-typical four-action cases generated with three actions, plus two source-typical three-action cases generated with two. Count differences alone are not a reason to pad lists. IDs49 and52 still address their requests; ID57 merits delivery clarification.','',
        '## Coherence review','',f'Counts: {counts}. Pass means relevant and internally coherent under this limited rubric. Unsure requires clarification; Issue identifies a concrete concern.','',
        'The clearest example is ID42: payment is already completed in history, yet the actions process it again. Other concerns include ignoring an explicitly named policy/context mismatch and omitting the requested deductible information. These need workflow/content improvements rather than a fixed action-count quota.','',
        '## Decision','',
        'Keep the prefix policy as a source-fidelity development candidate, with all regressions visible. Do not pad/replace actions with canonical source labels to improve a histogram. Do not start broad retraining based on this marginal metric alone. Keep existing weights and live defaults unchanged; expose the assessment and reviewed examples separately in the demo. A future targeted training experiment should pair coherent contexts/histories with correct workflows, then check both held-out workflow coherence and source-distribution drift. It remains future work, not a completed improvement.','',
        '| ID | Verdict | Observation |','|---|---|---|']
    lines += [f'| {w["sample_id"]} | {w["verdict"]} | {w["reason"]} |' for w in judgments]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(summary)


if __name__=='__main__':main()
