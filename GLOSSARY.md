# Drug Information Assistant

A chatbot that answers pharmacy staff and doctors about drugs strictly from an approved knowledge base, cites its sources, and declines or escalates everything else.

## Knowledge base

**Knowledge Base**:
The complete, approved set of nine KB Documents; the only source of facts the assistant may state.
_Avoid_: corpus, dataset, data

**KB Document**:
One approved document in the Knowledge Base, identified by its KB ID (`KB-01` … `KB-09`) and covering one or two diseases.
_Avoid_: file, doc, source file

**Section**:
A drug-class block within a KB Document, under a disease heading (e.g. "COPD › SABA"), listing products, uses, side effects and contraindications.
_Avoid_: chunk, part

**Citation**:
A pointer from an answer to the KB Document, Section and verbatim snippet that supports it.
_Avoid_: reference, source link

## Drugs

**Trade Name**:
The brand under which a product is sold in Egypt (e.g. Controloc).
_Avoid_: brand name, commercial name, product name

**Active Ingredient**:
The pharmacologically active substance(s) in a product (e.g. Pantoprazole).
_Avoid_: generic, active substance, molecule

**Drug Class**:
A pharmacological group of active ingredients (e.g. PPIs, ARBs).
_Avoid_: category, family, group

**Alternative**:
Another Trade Name with the same Active Ingredient(s). A different drug in the same Drug Class is not an Alternative; choosing one would be a Treatment Decision.
_Avoid_: substitute, equivalent, بديل علاجي

**Strength**:
The amount of an Active Ingredient in one unit of a product, as written in the Knowledge Base (e.g. Paracetamol 500 mg).
_Avoid_: concentration, dosage

**Dose**:
How much of a drug to give, how often, or to whom. Never stated by the assistant.
_Avoid_: dosage, posology, regimen

## Conversation outcomes

**Answered**:
The Knowledge Base supports what was asked, or part of it, and the reply carries at least one Citation.

**Out of Scope**:
Nothing that was asked is covered by the Knowledge Base, or the question is not about drug information at all.
_Avoid_: rejected, unanswerable, refused

**Escalated**:
The question is an Emergency or asks for a Dose, a Diagnosis or a Treatment Decision; the reply says what to do next without giving medical advice. Takes precedence over Answered and Out of Scope.
_Avoid_: refused, blocked, handed off

**Emergency**:
A message describing red-flag symptoms or an acutely unwell person (e.g. severe breathlessness with blue lips), regardless of whether the drug mentioned is in the Knowledge Base.

**Diagnosis**:
A request to decide what condition a specific person has.

**Treatment Decision**:
A request to choose, switch, start or stop therapy for a specific person. A question about what the Knowledge Base says for a group of patients (e.g. NSAIDs in pregnancy) is not a Treatment Decision.

**Session**:
One conversation thread; its earlier turns only resolve references like "it", and are never a source of facts.

**Answer Language**:
The language of the reply: Arabic when the user writes Arabic or dialect, English when they write English, and Mixed only for Franco-Arabic. Drug names stay in English and do not make an answer Mixed.
