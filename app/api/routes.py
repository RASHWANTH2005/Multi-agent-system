from fastapi import APIRouter, HTTPException

from app.schemas.research_schema import ResearchRequest, ResearchResponse
from app.services.research_service import run_research

router = APIRouter()


@router.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):

    try:
        report = await run_research(request.topic)

        return ResearchResponse(
            topic=request.topic,
            report=report
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )