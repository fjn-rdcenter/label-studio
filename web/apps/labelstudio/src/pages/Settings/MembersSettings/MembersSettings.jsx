import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "../../../components";
import { Label, Select } from "../../../components/Form";
import { confirm } from "../../../components/Modal/Modal";
import { Space } from "../../../components/Space/Space";
import { Spinner } from "../../../components/Spinner/Spinner";
import { useAPI } from "../../../providers/ApiProvider";
import { useProject } from "../../../providers/ProjectProvider";
import { Block, Elem } from "../../../utils/bem";
import "./MembersSettings.styl";

const ROLE_OPTIONS = [
  { value: "MA", label: "Manager" },
  { value: "RE", label: "Reviewer" },
  { value: "AN", label: "Annotator" },
];

export const MembersSettings = () => {
  const api = useAPI();
  const { project } = useProject();
  const callApi = api.callApi;
  const canManageMembers = project.can_manage_members === true;
  const [members, setMembers] = useState(null);
  const [orgUsers, setOrgUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState("");
  const [busy, setBusy] = useState(false);

  const fetchMembers = useCallback(async () => {
    if (!project.id) return;
    if (!canManageMembers) {
      setMembers([]);
      setOrgUsers([]);
      return;
    }

    const response = await callApi("projectMembers", {
      params: { pk: project.id },
      errorFilter: (result) => result.status === 403,
    });

    setMembers(Array.isArray(response) ? response : []);
  }, [callApi, canManageMembers, project.id]);

  const fetchOrgUsers = useCallback(async () => {
    if (!canManageMembers) return;

    const currentUser = await callApi("me");
    const organizationId = currentUser?.active_organization;
    if (!organizationId) return;

    const response = await callApi("memberships", {
      params: { pk: organizationId, page_size: 1000 },
      errorFilter: (result) => result.status === 403,
    });

    if (response?.results) setOrgUsers(response.results.map(({ user }) => user));
    else setOrgUsers([]);
  }, [callApi, canManageMembers]);

  useEffect(() => {
    fetchMembers();
    fetchOrgUsers();
  }, [fetchMembers, fetchOrgUsers]);

  // exclude users who are already assigned an explicit project role
  const availableUsers = useMemo(
    () => orgUsers.filter((user) => !members?.some((member) => member.user.id === user.id)),
    [orgUsers, members],
  );

  const addMember = useCallback(async () => {
    if (!selectedUserId) return;

    setBusy(true);
    await callApi("addProjectMember", {
      params: { pk: project.id },
      body: { user_id: Number(selectedUserId), role: "AN" },
    });
    setSelectedUserId("");
    setBusy(false);
    await fetchMembers();
  }, [callApi, project.id, selectedUserId, fetchMembers]);

  const changeRole = useCallback(
    async (memberId, role) => {
      await callApi("updateProjectMember", {
        params: { pk: project.id, memberID: memberId },
        body: { role },
      });
      await fetchMembers();
    },
    [callApi, project.id, fetchMembers],
  );

  const removeMember = useCallback(
    (member) => {
      confirm({
        title: "Remove member",
        body: `Remove ${member.user.email} from this project?`,
        okText: "Remove",
        buttonLook: "destructive",
        onOk: async () => {
          await callApi("deleteProjectMember", { params: { pk: project.id, memberID: member.id } });
          await fetchMembers();
        },
      });
    },
    [callApi, project.id, fetchMembers],
  );

  return (
    <Block name="members-settings">
      <h1>Members</h1>
      <Label description="Assign Manager, Reviewer, or Annotator access for this project." />

      {canManageMembers ? (
        <Elem name="controls">
          <Space>
            <div style={{ display: "flex", gap: "328px" }}>
              <Select
                placeholder="Select a user to add..."
                value={selectedUserId}
                onChange={(e) => setSelectedUserId(e.target.value)}
                options={availableUsers.map((user) => ({ value: String(user.id), label: user.email }))}
                style={{ width: 320 }}
              />
              <Button primary disabled={!selectedUserId || busy} waiting={busy} onClick={addMember}>
                Add Member
              </Button>
            </div>
          </Space>
        </Elem>
      ) : null}

      {!canManageMembers ? (
        <Elem name="empty">Only project Managers and organization Admins can manage project members.</Elem>
      ) : members === null ? (
        <div style={{ display: "flex", justifyContent: "center", marginTop: 32 }}>
          <Spinner size={32} />
        </div>
      ) : (
        <Elem name="list">
          {members.map((member) => (
            <Elem key={member.id} name="item">
              <Elem name="email">{member.user.email}</Elem>
              <Elem name="role">
                <Select
                  value={member.role}
                  options={ROLE_OPTIONS}
                  onChange={(e) => changeRole(member.id, e.target.value)}
                />
              </Elem>
              <Button look="danger" onClick={() => removeMember(member)}>
                Remove
              </Button>
            </Elem>
          ))}
          {members.length === 0 && <Elem name="empty">No members assigned yet. All organization members have Annotator access by default.</Elem>}
        </Elem>
      )}
    </Block>
  );
};

MembersSettings.title = "Members";
MembersSettings.path = "/members";
