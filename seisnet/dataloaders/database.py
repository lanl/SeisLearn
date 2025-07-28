from sqlalchemy import create_engine, Column, Integer, String, Float, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

from seisnet.utils import get_repo_dir
from pathlib import Path
from loguru import logger

Base = declarative_base()

class CorrelationTable(Base):
    __tablename__ = "correlations"
    id = Column(Integer, primary_key=True)
    waveform_id_1 = Column(String)
    waveform_id_2 = Column(String)
    correlation = Column(Float)

    __table_args__ = (UniqueConstraint("waveform_id_1", "waveform_id_2", name="_waveform_pair_uc"),)

    def __init__(self, waveId1, waveId2, corr_coef):
        self.waveform_id_1 = waveId1
        self.waveform_id_2 = waveId2
        self.correlation = corr_coef
    
    def insert(self, session, verbose=True):
        try:
            session.add(self)
            session.commit()
            return self.id
        except Exception as e:
            if verbose:
                logger.info(f"Error inserting entry: {e}")
            session.rollback()
            raise
    
    def update(self, session, verbose=True):
        try:
            session.commit()
            return self.id
        except Exception as e:
            if verbose:
                logger.info(f"Error updating entry: {e}")
            session.rollback()
            raise
    
    def delete(self, session, verbose=True):
        try:
            session.delete(self)
            session.commit()
        except Exception as e:
            if verbose:
                logger.info(f"Error deleting entry: {e}")
            session.rollback()
            raise


def create_local_session(db_path=None, return_engine=False):
    """
    Create a local session connected to the db
    """
    if not db_path:
        db_path = f"sqlite:///{Path(get_repo_dir())}//waveforms.db"
    
    engine, session = init_db(db_path)
    if return_engine:
        return engine, session
    else:
        return session


def init_db(db_path):
    """SQLite setup for local db"""
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    Session = scoped_session(session_factory)
    return engine, Session
