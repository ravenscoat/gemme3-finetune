"""Persist source-grounding diagnostic judgments; base arm is called grounded in raw schema."""
from dataset_catalog import read_rows,read_json
from insuranceqa_base_diagnostic import OUT,PRIOR
import review_insuranceqa_one_shot as reviewer
from json_io import write_json

NOTES={
0:('Pass','Directly answers territorial-coverage question.','Pass','Paraphrases the supplied US/Canada/Mexico territorial limits; real coverage not verified.'),
1:('Pass','Directly answers switching during a claim with a qualification.','Pass','Preserves freedom to switch and early-cancellation advice from source.'),
2:('Pass','Clearly identifies all three benefit types.','Pass','Benefits and eligibility descriptions follow the source.'),
3:('Pass','Answers premium-allocation question.','Pass','Uses fixed/indexed allocation and nonnegative-return source descriptions; real policy guarantees unverified.'),
4:('Pass','Directly answers creditor/beneficiary question with conditions.','Pass','Preserves personal-beneficiary ownership and creditor-claim qualifications.'),
5:('Pass','Directly answers preparation question; real medical/underwriting advice unverified.','Pass','Restates source abstinence timing and application/MIB claims; support is not endorsement.'),
6:('Pass','Coherently explains a possible monthly-versus-longer cost difference.','Issue','Adds administrative-cost causation not supplied in the passage.'),
7:('Pass','Directly explains deductible/premium cost expectations.','Pass','Source factors and forecasts are paraphrased; historical2014scope remains a limitation.'),
8:('Pass','Directly answers insurance amount and records needed.','Pass','Preserves rebuild-not-resale value and belongings/inventory advice.'),
9:('Issue','Confuses misrepresentation at application with starting smoking afterward.','Issue','Contradicts source no-penalty-after-purchase condition and generalizes one state to many.'),
10:('Pass','Answers employer-versus-driver responsibilities.','Pass','Preserves commercial vehicle, employer and driver-list relationships.'),
11:('Pass','Clearly answers employment and occupation distinction.','Pass','Paraphrases source without a new consequence.'),
12:('Pass','Answers other AAA product types.','Issue','Drops Texas scope and source uncertainty about other local offices, generalizing availability.'),
13:('Pass','Answers cost/eligibility question using source conditions; real ACA accuracy unverified.','Pass','Restates source poverty/uninsurable and means-testing assertions; dated scope remains a limitation.'),
14:('Issue','Assigns80percent payment to non-necessary therapy contrary to its medical-necessity condition.','Issue','Reverses the source non-necessary-services exclusion and misplaces the cap/necessity relationship.'),
15:('Pass','Explains bonus incentive and recapture drawback.','Pass','Retains first-year bonus and conditional forfeiture without added product identity.'),
16:('Issue','Blurs roof-rain damage with groundwater and drainage failure in the requested groundwater explanation.','Issue','Changes the source distinctions and omits the positive conditional drainage-endorsement coverage.'),
17:('Pass','Clearly distinguishes supplement and Advantage change timing.','Pass','Preserves source enrollment-period distinctions.'),
18:('Pass','Gives a coherent pregnancy timing answer; current underwriting truth unverified.','Issue','Strengthens one author\'s awareness into an almost-all-carriers current claim, losing the source epistemic qualification.'),
19:('Pass','Answers with examples and access/cost-control explanation; healthcare classification truth unverified.','Issue','Makes the VA handle military and public healthcare rather than preserving the separate source list.'),
20:('Pass','Answers suitability factors for variable-life account choices.','Pass','Preserves premiums/benefits, beneficiary interests, differing risks and broker role.'),
21:('Pass','Directly identifies longevity and elder-care costs.','Pass','Paraphrases source risk and care-cost relationship.'),
22:('Pass','Explains executive plan purpose and vesting incentive.','Pass','Preserves qualified-plan-limit and retention conditions.'),
23:('Pass','Answers deduction and benefit-taxability question conditionally.','Pass','Retains business qualification and ordinary-income benefit treatment; actual tax correctness unverified.'),
24:('Pass','Answers circumstances for dropping cover with caution.','Issue','Omits source no-debt qualification on wealth and states re-underwriting as a requirement rather than a possible subsequent change.'),
25:('Pass','Answers original purpose and later expansion.','Pass','Preserves source older-person/disability-group descriptions.'),
26:('Pass','Directly explains financial-risk protection.','Pass','Paraphrases source unexpected illness/accident rationale.'),
27:('Issue','Asks about renters but answers homeowners, inheriting source-domain inconsistency.','Pass','Answer preserves the source homeowners money/securities statements; source-label inconsistency is separate.'),
28:('Pass','Explains investment use and retirement borrowing.','Pass','Preserves premium-funding/MEC condition, loans and term-policy distinction.'),
29:('Pass','Clearly explains internal funding, administrator and stop-loss arrangements.','Pass','Paraphrases the source financing and claims-management relationships.'),
30:('Pass','Directly answers first-time application route; address/current process not verified.','Pass','Uses the source first-enrollment instructions without introducing the replacement-card typo.'),
31:('Pass','Explains fixed interest and inflation/fee considerations.','Pass','Preserves source accumulation, fees, inflation and agent advice.')}

def main():
    reviewer.OUT=OUT;reviewer.PRIOR=PRIOR;reviewer.NOTES=NOTES;reviewer.main()
    result=read_json(OUT/'review_summary.json');result['arm_names']={'baseline':'adapter','grounded':'base'}
    for key in ['demonstration_topic_lexical_flags','topic_flag_scope']:result.pop(key,None)
    result['markdown_fence_outputs']={arm:sum(r['text'].strip().startswith('```') for r in read_rows(OUT/(arm+'_generations.jsonl'))) for arm in ['baseline','grounded']}
    result['format_scope']='Existing parser accepts fenced JSON; fenced output still violates the no-markdown prompt instruction. Report separately from schema validity.'
    write_json(OUT/'review_summary.json',result)

if __name__=='__main__':main()
