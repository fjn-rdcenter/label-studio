import json

import pytest
import requests_mock
from projects.models import ProjectMember
from projects.models import Task
from rest_framework import status
from rest_framework.test import APIClient

from label_studio.tests.utils import make_project, register_ml_backend_mock

ORIG_MODEL_NAME = 'basic_ml_backend'
PROJECT_CONFIG = """<View><Image name="image" value="$image_url"/><Choices name="label"
          toName="image"><Choice value="pos"/><Choice value="neg"/></Choices></View>"""


@pytest.mark.django_db
@pytest.mark.parametrize('role', [ProjectMember.Role.ANNOTATOR, ProjectMember.Role.REVIEWER])
def test_labelers_can_open_project_without_reading_ml_backend_configuration(annotator_client, configured_project, role):
    membership = ProjectMember.objects.get(user=annotator_client.user, project=configured_project)
    membership.role = role
    membership.save(update_fields=['role'])
    backend = configured_project.ml_backends.first()
    client = APIClient()
    client.force_authenticate(user=annotator_client.user)

    list_response = client.get(f'/api/ml/?project={configured_project.id}')
    assert list_response.status_code == 200
    backend.refresh_from_db()  # the list call refreshes backend state
    assert list_response.data == [
        {
            'id': backend.id,
            'is_interactive': backend.is_interactive,
            'title': backend.title,
            'state': backend.state,
            'model_version': backend.model_version,
        }
    ]
    assert client.get(f'/api/ml/{backend.id}').status_code == 403
    assert client.post(
        '/api/ml/',
        {'project': configured_project.id, 'title': 'unauthorized', 'url': backend.url},
        format='json',
    ).status_code == 403
    assert client.post(f'/api/ml/{backend.id}/train').status_code == 403
    assert client.post(f'/api/ml/{backend.id}/predict/test?random=true').status_code == 403


@pytest.mark.django_db
def test_labeler_task_prediction_does_not_change_project_model_version(annotator_client, configured_project):
    backend = configured_project.ml_backends.first()
    task = configured_project.tasks.first()
    version_before = configured_project.model_version
    result = [{'from_name': 'text_class', 'to_name': 'text', 'type': 'choices', 'value': {'choices': ['class_A']}}]
    client = APIClient()
    client.force_authenticate(user=annotator_client.user)
    with requests_mock.Mocker() as m:
        register_ml_backend_mock(
            m,
            url=backend.url,
            setup_model_version='tasdf-v9',
            predictions={'results': [{'result': result, 'score': 0.5, 'model_version': 'tasdf-v9'}]},
        )
        response = client.post(f'/api/ml/{backend.id}/predict/test?task={task.id}')
        # Auto-Annotation re-asks on every task open: same model version -> same prediction, no new row
        again = client.post(f'/api/ml/{backend.id}/predict/test?task={task.id}')
    assert response.status_code == 200, response.data
    assert response.data['data']['model_version'] == 'tasdf-v9'
    assert again.data['data']['id'] == response.data['data']['id']
    assert task.predictions.filter(model_version='tasdf-v9').count() == 1
    configured_project.refresh_from_db()
    assert configured_project.model_version == version_before


@pytest.mark.django_db
def test_task_open_skips_backend_when_backend_version_prediction_exists(business_client, configured_project):
    from tasks.models import Prediction

    backend = configured_project.ml_backends.first()
    backend.model_version = 'tasdf-v9'
    backend.save(update_fields=['model_version'])
    configured_project.model_version = 'some-other-version'  # never returned by the backend
    configured_project.show_collab_predictions = True
    configured_project.save(update_fields=['model_version', 'show_collab_predictions'])
    task = configured_project.tasks.first()
    Prediction.objects.create(task=task, project=configured_project, result=[], model_version='tasdf-v9')
    with requests_mock.Mocker() as m:  # nothing registered: any call to the ML backend would raise
        response = business_client.get(f'/api/tasks/{task.id}')
    assert response.status_code == 200
    assert m.call_count == 0


@pytest.fixture
def ml_backend_for_test_api(ml_backend):
    register_ml_backend_mock(
        ml_backend,
        url='https://ml_backend_for_test_api',
        setup_model_version='1.0.0',
    )
    yield ml_backend


@pytest.fixture
def mock_gethostbyname(mocker):
    mocker.patch('socket.gethostbyname', return_value='321.21.21.21')


@pytest.mark.django_db
def test_ml_backend_set_for_prelabeling(business_client, ml_backend_for_test_api, mock_gethostbyname):
    project = make_project(
        config=dict(
            is_published=True,
            label_config=PROJECT_CONFIG,
            title='test_ml_backend_creation',
        ),
        user=business_client.user,
    )

    assert project.model_version == ''

    # create ML backend
    response = business_client.post(
        '/api/ml/',
        data={
            'project': project.id,
            'title': 'ml_backend_title',
            'url': 'https://ml_backend_for_test_api',
        },
    )
    assert response.status_code == 201

    project.refresh_from_db()
    assert project.model_version == 'ml_backend_title'


@pytest.mark.django_db
def test_ml_backend_not_set_for_prelabeling(business_client, ml_backend_for_test_api, mock_gethostbyname):
    """We are not setting it when its already set for another name,
    for example when predictions were uploaded before"""

    project = make_project(
        config=dict(
            is_published=True,
            label_config=PROJECT_CONFIG,
            title='test_ml_backend_creation',
        ),
        user=business_client.user,
    )

    project.model_version = ORIG_MODEL_NAME
    project.save()

    # create ML backend
    response = business_client.post(
        '/api/ml/',
        data={
            'project': project.id,
            'title': 'ml_backend_title',
            'url': 'https://ml_backend_for_test_api',
        },
    )
    assert response.status_code == 201

    project.refresh_from_db()
    assert project.model_version == ORIG_MODEL_NAME


@pytest.mark.django_db
def test_model_version_on_save(business_client, ml_backend_for_test_api, mock_gethostbyname):
    project = make_project(
        config=dict(
            is_published=True,
            label_config=PROJECT_CONFIG,
            title='test_ml_backend_creation',
        ),
        user=business_client.user,
    )

    assert project.model_version == ''

    # create ML backend
    response = business_client.post(
        '/api/ml/',
        data={
            'project': project.id,
            'title': 'test_ml_backend_creation_ML_backend',
            'url': 'https://ml_backend_for_test_api',
        },
    )
    assert response.status_code == 201
    r = response.json()
    ml_backend_id = r['id']
    response = business_client.get(f'/api/ml/{ml_backend_id}')
    assert response.status_code == 200
    assert response.json()['state'] == 'CO'

    # select model version in project
    assert (
        business_client.patch(
            f'/api/projects/{project.id}',
            data=json.dumps({'model_version': 'test_ml_backend_creation_ML_backend'}),
            content_type='application/json',
        ).status_code
        == 200
    )

    # change ML backend title --> model version should be updated
    assert (
        business_client.patch(
            f'/api/ml/{ml_backend_id}',
            data=json.dumps(
                {
                    'project': project.id,
                    'title': 'new_title',
                    'url': 'https://ml_backend_for_test_api',
                }
            ),
            content_type='application/json',
        ).status_code
        == 200
    )
    project.refresh_from_db()
    assert project.model_version == 'new_title'


@pytest.mark.django_db
def test_model_version_on_delete(business_client, ml_backend_for_test_api, mock_gethostbyname):
    project = make_project(
        config=dict(
            is_published=True,
            label_config=PROJECT_CONFIG,
            title='test_ml_backend_creation',
        ),
        user=business_client.user,
    )

    assert project.model_version == ''

    # create ML backend
    response = business_client.post(
        '/api/ml/',
        data={
            'project': project.id,
            'title': 'test_ml_backend_creation_ML_backend',
            'url': 'https://ml_backend_for_test_api',
        },
    )
    assert response.status_code == 201
    r = response.json()
    ml_backend_id = r['id']
    response = business_client.get(f'/api/ml/{ml_backend_id}')
    assert response.status_code == 200
    assert response.json()['state'] == 'CO'

    # select model version in project
    assert (
        business_client.patch(
            f'/api/projects/{project.id}',
            data=json.dumps({'model_version': 'test_ml_backend_creation_ML_backend'}),
            content_type='application/json',
        ).status_code
        == 200
    )

    project.refresh_from_db()
    assert project.model_version == 'test_ml_backend_creation_ML_backend'

    # delete ML backend --> project's model version should be reset
    assert business_client.delete(f'/api/ml/{ml_backend_id}').status_code == 204
    project.refresh_from_db()
    assert project.model_version == ''


@pytest.mark.django_db
def test_security_write_only_payload(business_client, ml_backend_for_test_api, mock_gethostbyname):
    project = make_project(
        config=dict(
            is_published=True,
            label_config=PROJECT_CONFIG,
            title='test_ml_backend_creation',
        ),
        user=business_client.user,
    )

    # create ML backend - fails without password
    response = business_client.post(
        '/api/ml/',
        data={
            'project': project.id,
            'title': 'test_ml_backend_creation_ML_backend',
            'url': 'https://ml_backend_for_test_api',
            'auth_method': 'BASIC_AUTH',
            # 'basic_auth_user': 'user',
            # 'basic_auth_pass': '<SECRET>',
        },
    )
    assert response.status_code == 400
    r = response.json()
    assert (
        r['validation_errors']['non_field_errors'][0]
        == 'Authentication username and password is required for Basic Authentication.'
    )

    # create ML backend with username and password
    response = business_client.post(
        '/api/ml/',
        data={
            'project': project.id,
            'title': 'test_ml_backend_creation_ML_backend',
            'url': 'https://ml_backend_for_test_api',
            'auth_method': 'BASIC_AUTH',
            'basic_auth_user': 'user',
            'basic_auth_pass': '<SECRET>',
        },
    )
    assert response.status_code == 201
    r = response.json()
    # security check that password is not returned in POST response
    assert 'basic_auth_pass' not in r
    ml_backend_id = r['id']
    response = business_client.get(f'/api/ml/{ml_backend_id}')
    assert response.status_code == 200
    # check that password is not returned in GET response
    assert 'basic_auth_pass' not in response.json()

    # patch ML backend without password - must pass since it uses write_only field for previous password
    response = business_client.patch(
        f'/api/ml/{ml_backend_id}',
        data=json.dumps(
            {
                'project': project.id,
                'title': 'new_title_1',
                'url': 'https://ml_backend_for_test_api',
            }
        ),
        content_type='application/json',
    )
    assert response.status_code == 200
    # check that password is not returned in PATCH response
    assert 'basic_auth_pass' not in response.json()

    # patch ML backend with password
    response = business_client.patch(
        f'/api/ml/{ml_backend_id}',
        data=json.dumps(
            {
                'project': project.id,
                'title': 'new_title',
                'url': 'https://ml_backend_for_test_api',
                'basic_auth_pass': '<ANOTHER_SECRET>',
            }
        ),
        content_type='application/json',
    )
    # check that password is not returned in PATCH response
    assert 'basic_auth_pass' not in response.json()

    from ml.models import MLBackend

    ml_backend = MLBackend.objects.get(id=ml_backend_id)
    assert ml_backend.basic_auth_pass == '<ANOTHER_SECRET>'


@pytest.mark.django_db
def test_ml_backend_predict_test_api_post_random_true(business_client):
    project = make_project(
        config=dict(
            is_published=True,
            label_config=PROJECT_CONFIG,
            title='test_ml_backend_creation',
        ),
        user=business_client.user,
        use_ml_backend=True,
    )
    Task.objects.create(project=project, data={'image': 'http://example.com/image.jpg'})

    # get ML backend id from project
    project.refresh_from_db()
    ml_backend = project.get_ml_backends().first()

    response = business_client.post(f'/api/ml/{ml_backend.id}/predict/test?random=true')

    assert response.status_code == status.HTTP_200_OK
    r = response.json()
    assert r['url'] == 'http://localhost:8999/predict'
    assert r['status'] == 200


TIMELAPSE_CONFIG = """<View>
  <Repeater on="$timelapse" indexFlag="{{idx}}" mode="pagination">
    <Image name="timelapse_{{idx}}" valueList="$timelapse[{{idx}}].image_urls"/>
    <PolygonLabels name="pl-{{idx}}" toName="timelapse_{{idx}}"><Label value="ZP"/><Label value="Blastocoel"/></PolygonLabels>
  </Repeater>
  <Text name="text" value="note"/>
  <Choices name="pn" toName="text"><Choice value="Unknown"/></Choices>
</View>"""
TIMELAPSE_DATA = {
    'timelapse': [
        {'time': '4:13:17:56', 'image_urls': ['/data/local-files/?d=w/0/1.jpg', '/data/local-files/?d=w/0/2.jpg']},
        {'time': '4:13:32:56', 'image_urls': ['/data/local-files/?d=w/1/1.jpg', '/data/local-files/?d=w/1/2.jpg']},
    ]
}


def _timelapse_backend(business_client):
    project = make_project(
        config=dict(is_published=True, label_config=TIMELAPSE_CONFIG, title='timelapse'),
        user=business_client.user,
        use_ml_backend=True,
    )
    task = Task.objects.create(project=project, data=TIMELAPSE_DATA)
    return project.get_ml_backends().first(), task


@pytest.mark.django_db
def test_smart_prediction_sends_the_pointed_image_as_a_single_image(business_client):
    backend, task = _timelapse_backend(business_client)
    region = {'from_name': 'pl-1', 'to_name': 'timelapse_1', 'type': 'polygonlabels',
              'value': {'points': [[1, 1], [5, 1], [5, 5]], 'polygonlabels': ['ZP']}}
    url = '/data/local-files/?d=w/1/2.jpg'
    with requests_mock.Mocker() as m:
        register_ml_backend_mock(m, url=backend.url, predictions={'results': [{'result': [region], 'score': 0.9}]})
        response = business_client.post(
            f'/api/ml/{backend.id}/predict/test',
            data=json.dumps({'task': task.id, 'image': {'url': url, 'to_name': 'timelapse_1', 'item_index': 1}}),
            content_type='application/json',
        )
        sent = [r.json() for r in m.request_history if r.path == '/predict'][-1]
    assert response.status_code == 200, response.content
    assert sent['tasks'][0]['data'] == {'image': url}
    assert sent['tasks'][0]['id'] == task.id
    config = sent['label_config']
    assert '<Image name="timelapse_1" value="$image" />' in config
    assert 'name="pl-1"' in config and 'toName="timelapse_1"' in config and 'Blastocoel' in config
    assert '{{idx}}' not in config and 'Repeater' not in config and 'name="pn"' not in config
    assert response.json()['data']['result'][0]['item_index'] == 1


@pytest.mark.django_db
@pytest.mark.parametrize(
    'image',
    [
        {'url': 'http://evil.example/x.jpg', 'to_name': 'timelapse_0', 'item_index': 0},
        {'url': '/data/local-files/?d=w/0/1.jpg', 'to_name': 'timelapse_0', 'item_index': '0'},
        'not-a-dict',
    ],
)
def test_smart_prediction_rejects_an_image_outside_the_task(business_client, image):
    backend, task = _timelapse_backend(business_client)
    with requests_mock.Mocker() as m:
        register_ml_backend_mock(m, url=backend.url)
        response = business_client.post(
            f'/api/ml/{backend.id}/predict/test',
            data=json.dumps({'task': task.id, 'image': image}),
            content_type='application/json',
        )
        assert not [r for r in m.request_history if r.path == '/predict']
    assert response.status_code == 400


@pytest.mark.django_db
def test_backend_error_message_reaches_the_user(business_client):
    backend, task = _timelapse_backend(business_client)
    with requests_mock.Mocker() as m:
        register_ml_backend_mock(m, url=backend.url)
        m.post(f'{backend.url}/predict', status_code=400, json={'error': 'label config needs a PolygonLabels tag'})
        smart = business_client.post(
            f'/api/ml/{backend.id}/predict/test', data=json.dumps({'task': task.id}), content_type='application/json'
        )
        debug = business_client.post(f'/api/ml/{backend.id}/predict/test?random=true')
    assert smart.status_code == 200
    assert 'label config needs a PolygonLabels tag' in smart.json()['errors'][0]
    assert debug.status_code == 200  # was a 500: Response(**None)
    assert debug.json()['status'] == 400
    assert 'label config needs a PolygonLabels tag' in debug.json()['error_message']
