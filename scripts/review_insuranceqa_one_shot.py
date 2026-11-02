"""Save explicit assistant reviews and compare with frozen zero-shot judgments."""
from collections import Counter
import hashlib
import json
import re
from dataset_catalog import read_rows,read_json,signature,sha,ROOT
from insuranceqa_one_shot_pilot import OUT,PRIOR
from json_io import write_json

# Filled only after the assistant reads each actual output alongside its passage.
NOTES={
 0:('Pass','Directly answers the registration/coverage question; actual territorial rules not verified.','Issue','Passage does not establish the added rule that Canadian registration categorically prevents an American policy applying.'),
 1:('Unsure','Penalty explanation mixes deductible, non-renewal and refusal to reapply without a clear mechanism.','Issue','Adds deductible, non-renewal and reapplication consequences absent from the cancellation passage.'),
 2:('Pass','Directly defines the three named Social Security benefits.','Pass','Retains the supplied benefit descriptions and qualification; substantial wording overlap is separate.'),
 3:('Unsure','Cash-surrender explanation mixes guarantees, index performance and losses without resolving the relationships.','Issue','Adds surrender-value definition, whole-life comparison, charges and index mechanics absent from the passage.'),
 4:('Pass','Gives a direct beneficiary/creditor answer; current legal correctness not verified.','Issue','Turns a qualified protection for personal beneficiaries into an unconditional opposite-case creditor entitlement.'),
 5:('Unsure','Moves between drug detection, medical-card acceptance and exam waiver without a clear consistent explanation.','Issue','Adds smoking-only detection and medical-card/exam-waiver claims not in the passage.'),
 6:('Pass','Suggests rate comparison to find the cheapest insurer for the requester; no factual pricing validation.','Issue','Changes month-by-month coverage topic to under-25 shopping and introduces website/comparison details absent from the source.'),
 7:('Pass','Gives cost/utilization reasons for increasing deductibles; current applicability not verified.','Pass','Uses source cost, aging and utilization explanations, largely verbatim; historical scope and copying remain limitations.'),
 8:('Pass','Answers coverage-amount question using rebuilding and property considerations; concrete amounts unverified.','Issue','Adds specific square-foot prices, an example coverage amount, pools/theaters and agent-monitoring advice not supplied.'),
 9:('Unsure','Smoking-coverage explanation ends with reduced-face/no-smoking discounts without a clear relationship to the smoker.','Issue','Adds health-risk, nuisance and discount claims absent from the misrepresentation passage.'),
 10:('Issue','Personal-auto question receives a denial framed around bus insurance, conflating private-car and commercial-bus coverage.','Issue','Contradicts the source statement that drivers can insure their own cars under the usual requirements.'),
 11:('Pass','Directly answers employment-status question.','Pass','Copies the complete passage; support does not establish novelty or current factual validity.'),
 12:('Pass','Directly answers AAA renters question with a Texas/local-office qualification.','Pass','Almost verbatim passage with duplicated opening, identical to the old grounded output.'),
 13:('Issue','Question about expenses of being uninsurable is answered with enrollment penalty and subsidy statements.','Issue','Adds federal penalty and premium-tax-credit assertions absent from the passage.'),
 14:('Issue','Says Medicare pays nothing for non-necessary care but then assigns only20percent to the patient in the same condition.','Issue','Reattaches the covered-service cost-sharing rule to non-necessary treatment and omits the historical date.'),
 15:('Unsure','Describes payment mechanics without clearly explaining the incentive purpose asked about.','Issue','Adds premium reduction, a cash-payment framing and a specific3-5year recapture period absent from the passage.'),
 16:('Pass','Provides a coherent water-damage distinction with policy advice; actual coverage not verified.','Issue','Adds burst-pipe coverage and a general drainage exclusion instead of preserving the source conditional drainage coverage.'),
 17:('Issue','Explains election periods then adds an unclear doctor-only restriction and reason not to switch.','Issue','Adds doctor-based availability and switching advice absent from the passage.'),
 18:('Pass','Directly answers pregnancy timing question; real underwriting correctness unverified.','Pass','Copies the complete supplied passage; source support is not a current-truth certification.'),
 19:('Issue','VA question shifts to Medicaid/Medicare travel and payment details rather than a coherent VA service description.','Issue','Adds procedural payment, free-care and travel/lodging assertions not supplied.'),
 20:('Issue','Does not compare variable versus whole life and gives an unclear premium/death-benefit/no-value claim.','Issue','Adds whole-life renewal, premium accumulation and beneficiary-value assertions absent from the passage.'),
 21:('Pass','Directly answers whether long-term care insurance can help with costs; tax claim not verified.','Issue','Adds tax-free-income treatment not established by the source.'),
 22:('Pass','States taxation during distributions, answering the timing question.','Pass','Repeats source sentences and tax timing; copying remains a separate issue.'),
 23:('Pass','Answers who pays workplace versus individual premiums; actual benefit rules unverified.','Issue','Adds employer benefit and premium-payment rules absent from the tax-deduction passage.'),
 24:('Pass','Provides age/expiry-related premium explanation; insurance mechanics not verified.','Issue','Adds mortality and universal/variable-life interest/contract-charge explanations absent from the source.'),
 25:('Pass','Directly names the Medicare signatory and date; historical details not independently verified here.','Issue','Adds the Social Security1935date and Roosevelt name absent from the passage; unsupported does not mean false.'),
 26:('Pass','Gives a health-conditioned position and ACA discussion; correctness of that advice is not verified.','Issue','Replaces the source financial-risk rationale with a no-need-if-healthy rule and adds benefits/cost assertions.'),
 27:('Unsure','QA pair itself is coherent about homeowners cover, but its renters domain conflicts; this inconsistency is inherited from the supplied source.','Pass','Closely paraphrases the supplied money/securities and endorsement passage, preserving qualifications.'),
 28:('Unsure','Answers borrowing but shifts between MEC status and a non-qualified retirement account without explaining their relationship.','Issue','Moves MEC condition from premium funding to loans and adds retirement-account classification absent from the source.'),
 29:('Unsure','Does not give a clear self-insurance cost estimate; mixes employee cost with employer premium contributions.','Issue','Turns no participant experience difference into equal costs and adds Connecticut,50employees and30percent contribution figures.'),
 30:('Pass','Gives coverage/budget factors and a broker route for deciding on Medigap; actual eligibility not verified.','Issue','Medicare application/card-replacement passage does not establish Medigap decision rules.'),
 31:('Issue','Conflates inflation with an insurer-controlled variable-annuity feature and adds an unexplained current-rate assertion.','Issue','Adds broad safety, company inflation control and current low-rate claims absent from the passage.')}

def main():
    originals={r['sample_id']:r for r in read_rows(PRIOR/'judgments.jsonl') if r['arm']=='grounded'}
    records=read_rows(OUT/'grounded_records.jsonl');baseline=read_rows(OUT/'baseline_records.jsonl')
    assert set(NOTES)=={r['sample_id'] for r in records}
    judgments=[]
    for wrapper in baseline:
        j={**originals[wrapper['sample_id']],'arm':'baseline','review_origin':'Reused unchanged from prior source-grounded pilot'}
        assert j['record_sha256']==hashlib.sha256(signature(wrapper['record']).encode()).hexdigest()
        judgments.append(j)
    for r in records:
        verdict,reason,support,why=NOTES[r['sample_id']]
        judgments.append({'arm':'grounded','sample_id':r['sample_id'],'record_sha256':hashlib.sha256(signature(r['record']).encode()).hexdigest(),
          'meaning_verdict':verdict,'meaning_reason':reason,'source_support_verdict':support,'source_support_reason':why,
          'judge':'Assistant in this conversation, unblinded, not expert calibrated',
          'scope':'QA alignment/internal coherence and support by the supplied passage; not factual accuracy'})
    (OUT/'judgments.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in judgments),encoding='utf-8')
    summary={'meaning':{arm:dict(Counter(r['meaning_verdict'] for r in judgments if r['arm']==arm)) for arm in ['baseline','grounded']},
       'source_support':{arm:dict(Counter(r['source_support_verdict'] for r in judgments if r['arm']==arm)) for arm in ['baseline','grounded']},
       'all_valid_records_reviewed':True,'unblinded':True,'expert_calibrated':False,'baseline_judgments_reused':True}
    p=read_json(OUT/'protocol.json');matched={};leak={}
    for arm,rows in [('baseline',baseline),('grounded',records)]:
        matched[arm]=sum(r['record']['domain']==p['anchors'][r['sample_id']]['domain'] for r in rows)
        leak[arm]=[r['sample_id'] for r in rows if re.search(r'\btow(?:ing)?\b|\bengine fails\b|mechanical breakdown',r['record']['question']+' '+r['record']['answer'],re.I)
           and not re.search(r'\btow(?:ing)?\b|\bengine fails\b|mechanical breakdown',p['anchors'][r['sample_id']]['answer'],re.I)]
    summary['supplied_domain_matches']=matched
    summary['demonstration_topic_lexical_flags']=leak
    summary['topic_flag_scope']='Literal towing/engine-failure/mechanical-breakdown markers absent from the supplied anchor; flags are not proof of causation or a complete semantic leakage detector.'
    write_json(OUT/'review_summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
