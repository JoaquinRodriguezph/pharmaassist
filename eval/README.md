   # Evaluation set

   expected_route values:
   - rag: answer from knowledge-base documents, with citation
   - rag_unanswerable: search finds nothing relevant; must say so, no guessing
   - rag_then_refer: retrieve relevant text, then refuse personal advice and refer to a pharmacist
   - tool: call a tracker endpoint (read-only)
   - both: needs a tracker call and a document search
   - refuse: out of scope, medical, or injection; return the static refusal