import os

import pytest

import grass.script as gs


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """A GRASS session in a fresh XY (unprojected) project -- RRI's own
    'utm' config flag is left at 0 (latlon) by r.hydro.rri regardless of
    project type, so an XY project is sufficient for exercising the
    module's raster/STRDS-handling logic without needing a real CRS."""
    tmp_path = tmp_path_factory.mktemp("grass_session")
    project = "test_project"
    gs.create_project(tmp_path, project)
    with gs.setup.init(tmp_path / project, env=os.environ.copy()) as session:
        yield session
