"""Persist100 individually authored qualitative judgments after reading the full sample.

No judge model is executed here. Notes and labels below were written by the
assistant in this conversation, not inferred automatically from keywords.
"""
from collections import Counter
import hashlib
import json

from dataset_catalog import ROOT,read_json,read_rows,sha
from json_io import write_json

OUT=ROOT/'reports/advanced_analysis/v1'
JUDGMENTS={
'B01':('Pass','Increasing coverage matches upgrade_coverage; the reply walks through choosing and confirming an upgrade.'),
'B02':('Pass','The request to accept a settlement is answered with offer review, acceptance and verification steps.'),
'B03':('Pass','Checking health-insurance details agrees with the label and the policy-details navigation.'),
'B04':('Pass','An expiring home policy is addressed with renewal selection, review and payment if applicable.'),
'B05':('Pass','The request to contact the insurer is addressed through telephone, email or chat contact options.'),
'B06':('Pass','The request for an insurance representative is answered with representative contact and callback options.'),
'B07':('Pass','Life-insurance information is requested and the reply directs the user to life-policy details.'),
'B08':('Pass','The home-insurance information request is answered with policy-specific information navigation.'),
'B09':('Pass','The complaint request is matched by a complaint form, submission and reference-number follow-up.'),
'B10':('Pass','The payment-checking request is matched by instructions to view payment records.'),
'B11':('Pass','The user explicitly wants cancellation rather than renewal; the reply describes cancelling the selected policy.'),
'B12':('Pass','The life-policy coverage-change request is matched by selecting new coverage and confirming changes.'),
'B13':('Pass','Scheduling an appointment is matched by professional selection, a date/time and confirmation.'),
'B14':('Pass','Early termination fees agree with cancellation_fees and the reply locates policy-specific fee information.'),
'B15':('Unsure','The request says card without specifying its type; the reply assumes a debit card and gives debit-only instructions.'),
'B16':('Pass','Seeing different health policies is reasonably interpreted as comparison; the reply provides comparison-search steps.'),
'B17':('Pass','Receiving a payment matches the claims label and the reply selects a claim and payment method.'),
'B18':('Pass','The pet-insurance quote request is matched by pet details, coverage selection and a quote calculation.'),
'B19':('Pass','Checking a life-insurance claim status is matched by selecting the claim and viewing its status.'),
'B20':('Unsure','Submit some life insurance does not clearly identify a claim rather than an application; the file_claim interpretation is uncertain.'),
'B21':('Pass','Checking travel insurance is answered with navigation to travel-policy information.'),
'B22':('Pass','The pet-policy coverage question is answered by locating the selected policy coverage details.'),
'B23':('Pass','Reducing coverage agrees with downgrade_coverage; the repeated downgrade-menu wording is awkward but the action remains consistent.'),
'B24':('Pass','Viewing pet insurance is matched by retrieving the pet policy information.'),
'B25':('Pass','Viewing auto insurance is answered by selecting the auto policy under the account policies section.'),
'B26':('Pass','Changing personal details on a life policy is matched by editing, reviewing and submitting those details.'),
'B27':('Pass','Buying travel insurance is answered by selecting a travel plan and submitting the purchase details.'),
'B28':('Pass','Scheduling payments is matched by entering payment information, dates and confirmation.'),
'B29':('Pass','Checking life-insurance rates is matched by providing quote factors and viewing resulting rates.'),
'B30':('Pass','Rejecting an offer agrees with reject_settlement; the reply describes communicating refusal to the adjuster.'),
'B31':('Pass','Appealing an invoice agrees with dispute_invoice and the reply describes an invoice dispute submission.'),
'B32':('Pass','The explicit request for a human is answered with a representative contact option.'),
'B33':('Pass','Notifying an incident is matched by an incident report form and submission.'),
'B34':('Pass','General policy information is answered by browsing insurance offerings or contacting support.'),
'B35':('Pass','The settlement-negotiation request is answered with documentation review, adjuster contact and a revised offer.'),
'B36':('Pass','The failed transfer and payment-issue report request are matched by a payment issue reporting form.'),
'B37':('Pass','The customer-service contact request is answered with a support number and connection instructions.'),
'B38':('Pass','Upgrading health coverage is matched by selecting and confirming enhanced coverage.'),
'B39':('Pass','Accepting settlement offers is answered with offer review, acceptance and verification.'),
'B40':('Pass','Viewing health insurance is answered with health-policy information navigation.'),
'Q01':('Pass','The answer addresses life insurance and credit, but its assertions about premium payments and credit require factual verification.'),
'Q02':('Issue','The question asks which annuities are tax deferred; the reply repeats tax-related phrases without identifying any concrete qualifying type or distinction.'),
'Q03':('Unsure','The reply mentions comparing rates but gives no North Carolina-specific basis, and its single-company comparison sentence is unclear.'),
'Q04':('Pass','The reply directly challenges the no-online-quote premise and distinguishes a quote from an effective policy.'),
'Q05':('Pass','The answer addresses VA disability compensation and supplies a contact route; benefit eligibility and calculation claims are not verified.'),
'Q06':('Pass','The reply connects water-damage coverage to the cause and the specific policy; its anecdotal coverage claims are not verified.'),
'Q07':('Issue','The reply says Medicare is the only payer and denies secondary health insurers, then says another health insurer pays remaining claims; the explanation conflicts internally.'),
'Q08':('Issue','The question asks how many people are on Medicare; the reply gives eligibility information and mentions a report but provides no count or dated estimate.'),
'Q09':('Pass','The reply supplies a date in direct response to the enactment question. This is a relevance-only Pass; the separate official-source spot-check contradicts its1974 date.'),
'Q10':('Pass','The answer gives general shopping/agent advice rather than naming a universally cheapest provider; price and marketing assertions are not verified.'),
'Q11':('Pass','The reply directly says no and explains that annuity durations can differ, including lifetime arrangements.'),
'Q12':('Pass','The answer links renters-insurance price variation to area and product differences; its broad pricing assertions are unverified.'),
'Q13':('Issue','The answer starts by explaining a lower premium but its final proposed reason says the user is being offered higher rates; that reverses the question being explained.'),
'Q14':('Issue','The answer addresses Aetna disability insurance but its proposed website www.aetnacom is malformed, undermining the concrete contact instruction.'),
'Q15':('Unsure','The answer shifts from tree-removal costs to property damage and a neighbor car; the actual removal-coverage answer remains ambiguous and its exclusion claim is unverified.'),
'Q16':('Pass','Broker/carrier contact advice addresses the question. The ungrounded contact-me/profile persona and carrier availability are not verified.'),
'Q17':('Issue','The definition becomes incoherent at the sentence saying the best coverage one can get is the best one cannot get; plan and benefit assertions also need verification.'),
'Q18':('Pass','The answer gives retirement-plan examples and an explanation of qualified status; precise tax qualification is not verified.'),
'Q19':('Issue','The reply calls term life low cost but then says it generally costs much more than permanent coverage without explaining the conflicting framing; the numerical quote is also unsupported.'),
'Q20':('Issue','The question asks about fixed-annuity drawbacks; the answer shifts to defending variable annuities and does not clearly explain the requested fixed-annuity disadvantages.'),
'Q21':('Pass','The reply directly answers the bed-bug coverage question, but the blanket coverage assertion is unverified; relevance does not establish correctness.'),
'Q22':('Pass','The answer addresses teenager insurance shopping and comparison with a parent policy; its cheapest-option claim is not verified.'),
'Q23':('Pass','The reply defines disability insurance in terms of inability to work, benefits and duration, matching the question and domain.'),
'Q24':('Pass','The answer addresses home insurance and credit, while its vague credit-rating and fraud assertions require factual verification.'),
'Q25':('Unsure','Advantage plan is not clarified as Medicare Advantage or another named product; the broad HMO/network/referral description cannot be reliably resolved within the record.'),
'Q26':('Issue','The answer conflates a plan being in-network with providers accepting assignment, producing an unclear definition rather than a coherent explanation of the requested term.'),
'Q27':('Unsure','The undefined term creditor retirement plan and vague IRA-law explanation do not provide enough reliable context to judge the proposed definition.'),
'Q28':('Pass','The answer discusses life-insurance payment to an estate and its consequences, but the blanket beneficiary/default-estate assertions require independent verification.'),
'Q29':('Issue','The reply does not explain the Roth-IRA component or withdrawal conditions; it shifts into a named adviser promotion with concrete contact details and an unverified no-penalty assertion.'),
'Q30':('Issue','The question asks how a tenant buys renters insurance while on a lease; the reply shifts between landlord and tenant roles and gives no clear tenant purchase process.'),
'W01':('Issue','The context says the policy is active, but the workflow generates a cancellation-confirmation letter without checking whether cancellation occurred.'),
'W02':('Unsure','The message says the fire department is being called now; mitigation/inspection steps and high priority are plausible, but immediate-safety handling is not explicit.'),
'W03':('Pass','The workflow looks up the claim before providing a status and timeline; it does not visibly promise an unverified approval outcome.'),
'W04':('Unsure','Coverage verification is appropriate, but the required exclusion explanation presupposes an exclusion that is not established in the roof-leak conversation.'),
'W05':('Pass','The car-loss request and life-policy context suggest a policy mismatch; lookup and human handoff avoid simply asserting that the car is covered.'),
'W06':('Pass','The autopay request matches account verification, payment-information collection and enrollment as hypothetical expected actions.'),
'W07':('Issue','Investigating the quote/premium discrepancy is relevant, but the financial_hardship flag is not supported by any stated inability to pay or hardship in the input.'),
'W08':('Pass','The explicit bad-faith allegation is matched by documentation, legal-risk handling and human handoff.'),
'W09':('Unsure','A personal car-loss question is paired with a commercial-general-liability context; the workflow proposes an endorsement without clarifying the relevant policy.'),
'W10':('Unsure','A split-limit coverage request is paired with renters insurance; the workflow updates coverage without clarifying whether the requested limit structure fits that policy.'),
'W11':('Issue','The active-policy context conflicts with issuing a cancellation letter; identity verification alone does not resolve the cancellation-status discrepancy.'),
'W12':('Pass','The pre-existing damage statement is captured, documented and sent to a human/adjuster rather than automatically approved.'),
'W13':('Pass','The delayed-claim message is answered with lookup, status information, expectations and acknowledgment of frustration.'),
'W14':('Issue','The subcategory says renewal_and_change although no renewal is requested; the renters/new-house context also needs clarification before treating this as a routine endorsement.'),
'W15':('Unsure','The life-policy context and claimed homeowners emergency need reconciliation; new_policy_after_emergency and prevent_emergency_gaming are not clearly supported by the unrelated identity-theft request.'),
'W16':('Pass','Account lookup, late-fee calculation and a payment plan address the missed-payment problem, although future autopay prevention is not developed.'),
'W17':('Pass','The repeated unresolved contact history is matched by documenting failure, acknowledging frustration and handing off to a specialist.'),
'W18':('Pass','The regulator-complaint threat is matched by complaint documentation, high priority and supervisor handoff; no compliance certification is inferred.'),
'W19':('Pass','The renewal premium question is matched by policy lookup, rate-factor explanation and discount options.'),
'W20':('Pass','The new Nebraska address can legitimately differ from an old Missouri context; identity verification, update and premium recalculation address the request.'),
'W21':('Pass','The commercial-policy loss-run request is matched by identity verification, compilation and document delivery.'),
'W22':('Issue','The user explicitly asks whether they need to return to the ER; the workflow treats the whole exchange as standard-priority insurance processing and ignores that part of the request.'),
'W23':('Pass','The overdue-claim request is matched by claim lookup, a status update, expectation setting and acknowledgment.'),
'W24':('Unsure','The two auto-damage issues are collected and escalated, but the life-policy context is not explicitly reconciled with the auto workflow or verified in the actions.'),
'W25':('Issue','The context says active while the customer is uncertain about suspension; the workflow explains suspension/reinstatement without a policy-status verification step.'),
'W26':('Unsure','The user reports an already-delayed short refund; initiating a refund without an explicit trace of the prior payment may be inappropriate, although lookup/eligibility could resolve it.'),
'W27':('Pass','The system-freeze report is matched by acknowledgment, a workaround and issue reporting.'),
'W28':('Pass','The explicit request for a supervisor after an unresolved complaint is matched by complaint lookup and supervisor transfer.'),
'W29':('Unsure','The life-policy context does not establish that the customer means property-flood coverage; an exclusion/property-policy workflow assumes the coverage being discussed.'),
'W30':('Pass','The address-change request is matched by identity verification, address update and premium recalculation; Virginia can be the previous address.'),
}


def main():
    protocol=read_json(OUT/'protocol.json');records=read_rows(OUT/'review_records.jsonl')
    if sha(OUT/'review_records.jsonl')!=protocol['review']['input_sha256']:raise RuntimeError('Frozen review inputs changed')
    if set(JUDGMENTS)!={r['review_id'] for r in records}:raise RuntimeError('Every selected record must have one authored judgment')
    output=[];counts={};groups={}
    for record in records:
        verdict,note=JUDGMENTS[record['review_id']]
        output.append({'review_id':record['review_id'],'dataset':record['dataset'],'sample_id':record['sample_id'],'stratum':record['stratum'],
            'record_sha256':hashlib.sha256(json.dumps(record['record'],ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
            'verdict':verdict,'reason':note,'scope':'Relevance, label alignment, internal coherence; not factual accuracy',
            'fact_verification':'Not assessed by this rubric; see separate targeted spot-checks','judge':'Assistant in this conversation; exact model ID not independently recorded'})
    for key in ['bitext','insuranceqa','reliability']:
        rows=[r for r in output if r['dataset']==key];counts[key]={v:sum(r['verdict']==v for r in rows) for v in ['Pass','Issue','Unsure']}
        groups[key]=dict(Counter(r['stratum'] for r in rows))
    (OUT/'ai_judgments.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output),encoding='utf-8')
    summary={'n':len(output),'counts':counts,'strata':groups,'input_sha256':sha(OUT/'review_records.jsonl'),
             'rubric_sha256':sha(OUT/'rubric_v1.md'),'judgments_sha256':sha(OUT/'ai_judgments.jsonl'),
             'limits':['One unblinded AI reviewer; no human/expert calibration or inter-rater agreement.',
                       'Stratified descriptive sample, not a population accuracy estimate or base-vs-tuned comparison.',
                       'Pass means relevant/coherent within the rubric; factual assertions may still be wrong.',
                       'Workflow actions are hypothetical labels, not actual executed agent behavior.',
                       'Source inconsistencies may contribute; this review does not isolate their origin.']}
    write_json(OUT/'ai_review_summary.json',summary)
    lines=['# 100-record AI qualitative review','',
           '**Pass means relevance/coherence only. It does not certify factual insurance correctness.**','',
           '| Dataset | Pass | Issue | Unsure |','|---|---:|---:|---:|']
    for key,c in counts.items():lines.append(f"| {key} | {c['Pass']} | {c['Issue']} | {c['Unsure']} |")
    lines+=['',*['- '+v for v in summary['limits']],'','## All individual judgments','',
            '| Review ID / sample ID | Verdict | Reason |','|---|---|---|']
    for r in output:lines.append(f"| {r['review_id']} / {r['sample_id']} | {r['verdict']} | {r['reason'].replace('|','/')} |")
    lines+=['','Inputs, rubric, strata and hashes are saved alongside this report. Prior20-record review is preserved.','']
    (OUT/'ai_review_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
