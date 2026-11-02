"""Persist the conversation assistant's explicit qualitative judgments, not a classifier."""
from collections import Counter
import hashlib
from dataset_catalog import read_rows,read_json,signature,sha
from insuranceqa_final_pilot import OUT
from json_io import write_json

BASELINE=[
 ('Pass','Defines long-term disability in terms of income replacement after illness/injury.'),
 ('Pass','Addresses VA disability compensation; benefit calculation and contact assertions not verified.'),
 ('Issue','Question asks breast lift; answer substitutes enlargement/augmentation without resolving the distinction.'),
 ('Pass','Names plan types and explains qualified status; legal detail unverified.'),
 ('Unsure','Predicts a dated rate increase using an unclear investment-return explanation.'),
 ('Issue','Calls term life low cost and then more expensive than permanent coverage without explanation.'),
 ('Unsure','Gives a specific income threshold without date or household context; eligibility cannot be resolved.'),
 ('Unsure','Names a seller but gives an unclear most-popular waiting-period explanation.'),
 ('Issue','Calls term insurance non-expiring and contains confused product comparisons.'),
 ('Issue','Shifts landlord/tenant roles and does not give a coherent tenant purchase process.'),
 ('Pass','Addresses mold exclusions with policy-specific qualifications.'),
 ('Pass','Answers with a regulator discussion; agency names/current authority not verified.'),
 ('Pass','Explains comparing coverage and deductible levels, rather than price alone.'),
 ('Pass','Defines the deductible and gives exceptions; benefit rules unverified.'),
 ('Pass','Directly addresses physical therapy coverage with a medical-need qualification.'),
 ('Unsure','Undefined Advantage plan mixes network and referral assertions without clarifying the product.'),
 ('Issue','Says when needed but does not address when enrollment or purchase is permitted.'),
 ('Issue','Fixed-annuity drawbacks question shifts to variable-annuity promotion.'),
 ('Issue','Conflates the auto insurer with the rental company and gives an unclear coverage condition.'),
 ('Unsure','Lists price factors, then makes an unsupported broad annual-price assertion.'),
 ('Issue','Explains a lower premium by ending with being offered higher rates.'),
 ('Unsure','Water/flood examples are ambiguous; no clear covered-peril distinction.'),
 ('Issue','Says online buying is possible then describes needing an agency without explaining the apparent conflict.'),
 ('Pass','Suggests agents as a place to obtain renters insurance; price claims unverified.'),
 ('Issue','Vermont shopping question receives tangential state-tree/road commentary and no useful local route.'),
 ('Issue','Says Medicare is the only payer then describes another insurer paying claims.'),
 ('Unsure','Endowment/term/universal terminology is not clarified enough to resolve the definition.'),
 ('Issue','Denies insurance paying off cars but gives an unclear total-loss explanation without addressing the loan.'),
 ('Issue','Does not address Roth conditions; shifts to adviser promotion and unsupported penalty assertion.'),
 ('Issue','Normal maintenance and repair exclusions are described inconsistently rather than identifying landscaping cover.'),
 ('Pass','Gives an application/underwriting and broker route for disability insurance.'),
 ('Unsure','Acknowledges policy dependence but mixes towing situations without a clear coverage explanation.')]

GROUNDED=[
 ('Unsure','Unclear ownership/rental and US/Canada conditions make the direct answer hard to resolve.','Issue','Adds rental, ownership and company-contact conditions absent from the territorial passage.'),
 ('Pass','Explains an adjuster role with an awkward own-or-portion phrase.','Issue','Switching insurers passage does not support a claims-adjuster discussion.'),
 ('Pass','Explains retirement, survivor and disability benefits despite awkward question wording.','Pass','The three benefit explanations follow the passage; much wording is reused.'),
 ('Unsure','Gives a repetitive product explanation with unclear immediate-indexed terminology.','Issue','Adds immediate-indexed products, mutual-fund comparisons and broad policy-value assertions absent from the passage.'),
 ('Issue','Changes creditor, father, insured and estate roles without a coherent answer.','Issue','Adds creditor priority and estate-repayment claims and changes the passage beneficiary framing.'),
 ('Pass','Addresses how marijuana use can affect an application; real underwriting correctness unverified.','Issue','Adds a higher-rate outcome and broadens medical-use conditions absent from the passage.'),
 ('Pass','Answers shopping/rate variation with comparison advice.','Issue','Ignores monthly-policy source and adds truck/motorcycle pricing and discount claims.'),
 ('Pass','Gives reasons for a dated Medicare cost increase.','Issue','Adds a claimed replacement of medical care by a wellness industry; part of the remainder copies the passage.'),
 ('Issue','First says no insurance is needed, then says insurance is certainly needed.','Issue','Transfers the source no-added-resale-value answer to a different question and adds appreciation claims.'),
 ('Issue','Unclear smoker/non-smoker rate comparisons and cancellation narrative obscure the explanation.','Issue','Invents a 99-year contract and named sign-off, and changes the premium-return framing.'),
 ('Unsure','Gives ambiguous fault and firing conditions rather than a well-qualified answer.','Issue','Source says records matter to employment but does not establish firing/legal-liability rules.'),
 ('Pass','Directly answers unemployment and home-insurability question.','Pass','Answer reproduces the complete supplied passage; copying is recorded separately.'),
 ('Pass','Directly addresses AAA renters coverage, with duplicated opening phrase.','Pass','Reproduces almost all of the supplied answer, preserving Texas/local-office qualifications.'),
 ('Issue','Says rates do and do not vary by zip code and both excludes and includes health-history factors.','Issue','Adds many specific prices, locations and rating claims absent from the PPACA passage.'),
 ('Pass','Describes Part A and Part B; Medigap/Advantage relationship is not factually verified.','Issue','Physical-therapy passage does not support the added Part A/DME/Advantage descriptions.'),
 ('Issue','Question asks nonforfeitable annuities but answer defines bonus annuities and describes forfeiture.','Issue','Copies the bonus-annuity answer while changing the product identity, which the source does not support.'),
 ('Pass','Gives a coherent sudden/gradual water-damage distinction; policy accuracy unverified.','Issue','Adds burst-pipe coverage not in the passage and loses source qualifications on groundwater.'),
 ('Issue','Gives moving as an election-period example then says county availability means the customer would not qualify.','Issue','Adds county and eligibility claims that the passage does not establish.'),
 ('Pass','Answers the timing question with a coherent, but unverified, carrier-availability claim.','Pass','Reproduces the entire supplied passage; current underwriting truth is not established.'),
 ('Unsure','Unclear universal-access versus qualification statements and mixed VA/TRICARE conditions.','Issue','Adds taxes, universal coverage, laws, copays and TRICARE card entitlements absent from the source.'),
 ('Unsure','Mentions risk and suitability but does not clearly explain the bad-investment premise.','Pass','Answer is a near-verbatim source excerpt, including its beneficiary claim; truth unverified.'),
 ('Pass','Gives reasons and benefits for long-term care insurance; universal benefit/premium claims not verified.','Issue','Adds spouse cover, memory care, monthly benefit and guaranteed premium claims absent from the passage.'),
 ('Pass','Directly states accumulation and distribution tax treatment.','Pass','Answer consists of verbatim source sentences; copying is separate.'),
 ('Unsure','Question and premium-versus-benefit tax distinction remain ambiguous; broad individual-plan claim not resolved.','Issue','Introduces employer/individual distinctions and drops the source legitimate-business qualification.'),
 ('Pass','Gives circumstances for dropping coverage while recommending caution; numerical claim unverified.','Pass','Copies the complete source answer, including its numerical premium claim.'),
 ('Pass','Directly answers Medicare eligibility; conditions and contact detail not independently verified.','Issue','Adds Tennessee, ages, waiting periods, diagnoses and a phone number not supplied in the passage.'),
 ('Issue','Does not explain access help; shifts to a broad rating claim and vague debate commentary.','Issue','Introduces an unsupported historical first and rating-factor assertion.'),
 ('Unsure','Gives a speculative collision-claim explanation without sufficient context for the insurer relationship.','Issue','Money/securities passage provides no support for auto/renters collision-claim handling.'),
 ('Issue','Says no cap loss/gain and then says the policy is subject to a cap loss without resolving the contradiction.','Issue','Adds crediting/cap descriptions absent from the investment-funding passage.'),
 ('Unsure','Says true cost cannot be determined then recommends a model without explaining an estimating method.','Issue','Adds exclusions, inability to determine cost and financial-model claims absent from the passage.'),
 ('Issue','Replacement-card question receives a mixture of replacement and first-time enrollment instructions.','Pass','Largely repeats source instructions, including its unverified web-address typo; source support does not establish correctness.'),
 ('Issue','Shifts rate definition to an unexplained thousand-dollar versus five-thousand annual payment example.','Issue','Adds rate choice, lifetime guarantees and numerical payout claims absent from the passage.')]

def main():
    rows=[]
    for arm,notes in [('baseline',BASELINE),('grounded',GROUNDED)]:
        records=read_rows(OUT/(arm+'_records.jsonl'));assert len(records)==len(notes)
        for r,note in zip(records,notes):
            verdict,reason=note[:2]
            j={'arm':arm,'sample_id':r['sample_id'],'record_sha256':hashlib.sha256(signature(r['record']).encode()).hexdigest(),
               'meaning_verdict':verdict,'meaning_reason':reason,'scope':'Question-answer alignment and internal coherence; not factual accuracy',
               'judge':'Assistant in this conversation, unblinded, not expert calibrated'}
            if arm=='grounded':j.update(source_support_verdict=note[2],source_support_reason=note[3])
            rows.append(j)
    (OUT/'judgments.jsonl').write_text(''.join(__import__('json').dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    counts={a:dict(Counter(r['meaning_verdict'] for r in rows if r['arm']==a)) for a in ['baseline','grounded']}
    support=dict(Counter(r['source_support_verdict'] for r in rows if r['arm']=='grounded'))
    write_json(OUT/'review_summary.json',{'meaning':counts,'grounded_source_support':support,'all_valid_records_reviewed':True,'unblinded':True,'expert_calibrated':False})
    print(counts,support)

if __name__=='__main__':main()
