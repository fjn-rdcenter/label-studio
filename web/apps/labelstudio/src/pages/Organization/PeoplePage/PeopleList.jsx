import { formatDistance } from "date-fns";
import { useCallback, useEffect, useState } from "react";
import { Pagination, Spinner, Userpic } from "../../../components";
import { Select } from "../../../components/Form";
import { usePage, usePageSize } from "../../../components/Pagination/Pagination";
import { useAPI } from "../../../providers/ApiProvider";
import { Block, Elem } from "../../../utils/bem";
import { isDefined } from "../../../utils/helpers";
import { useUpdateEffect } from "../../../utils/hooks";
import "./PeopleList.styl";
import { CopyableTooltip } from "../../../components/CopyableTooltip/CopyableTooltip";

export const PeopleList = ({ onSelect, selectedUser, defaultSelected }) => {
  const api = useAPI();
  const [usersList, setUsersList] = useState();
  const [organizationId, setOrganizationId] = useState();
  const [organizationOwnerId, setOrganizationOwnerId] = useState();
  const [canManageRoles, setCanManageRoles] = useState(false);
  const [currentPage] = usePage("page", 1);
  const [currentPageSize] = usePageSize("page_size", 30);
  const [totalItems, setTotalItems] = useState(0);

  console.log({ currentPage, currentPageSize });

  const fetchUsers = useCallback(async (page, pageSize) => {
    const currentUser = await api.callApi("me");
    const orgId = currentUser?.active_organization;

    if (!orgId) return;

    setOrganizationId(orgId);
    const [organization, response, allMemberships] = await Promise.all([
      api.callApi("organization", { params: { pk: orgId } }),
      api.callApi("memberships", {
        params: {
          pk: orgId,
          contributed_to_projects: 1,
          page,
          page_size: pageSize,
        },
      }),
      api.callApi("memberships", { params: { pk: orgId, page_size: -1 } }),
    ]);

    if (response.results) {
      setUsersList(response.results);
      setTotalItems(response.count);
      setOrganizationOwnerId(organization?.created_by);
      const currentMembership = allMemberships?.results?.find(({ user }) => user.id === currentUser.id);
      setCanManageRoles(organization?.created_by === currentUser.id || currentMembership?.role === "AD");
    }
  }, []);

  const updateRole = useCallback(
    async (membership, role) => {
      await api.callApi("updateOrganizationMembership", {
        params: { pk: organizationId, user_pk: membership.user.id },
        body: { role },
      });
      await fetchUsers(currentPage, currentPageSize);
    },
    [api, currentPage, currentPageSize, fetchUsers, organizationId],
  );

  const selectUser = useCallback(
    (user) => {
      if (selectedUser?.id === user.id) {
        onSelect?.(null);
      } else {
        onSelect?.(user);
      }
    },
    [selectedUser],
  );

  useEffect(() => {
    fetchUsers(currentPage, currentPageSize);
  }, []);

  useEffect(() => {
    if (isDefined(defaultSelected) && usersList) {
      const selected = usersList.find(({ user }) => user.id === Number(defaultSelected));

      if (selected) selectUser(selected.user);
    }
  }, [usersList, defaultSelected]);

  return (
    <>
      <Block name="people-list">
        <Elem name="wrapper">
          {usersList ? (
            <Elem name="users">
              <Elem name="header">
                <Elem name="column" mix="avatar" />
                <Elem name="column" mix="email">
                  Email
                </Elem>
                <Elem name="column" mix="name">
                  Name
                </Elem>
                <Elem name="column" mix="last-activity">
                  Last Activity
                </Elem>
                <Elem name="column" mix="role">
                  Organization role
                </Elem>
              </Elem>
              <Elem name="body">
                {usersList.map((membership) => {
                  const { user } = membership;
                  const isOrganizationOwner = user.id === organizationOwnerId;
                  const role = isOrganizationOwner ? "AD" : membership.role;
                  const active = user.id === selectedUser?.id;

                  return (
                    <Elem key={`user-${user.id}`} name="user" mod={{ active }} onClick={() => selectUser(user)}>
                      <Elem name="field" mix="avatar">
                        <CopyableTooltip title={`User ID: ${user.id}`} textForCopy={user.id}>
                          <Userpic user={user} style={{ width: 28, height: 28 }} />
                        </CopyableTooltip>
                      </Elem>
                      <Elem name="field" mix="email">
                        {user.email}
                      </Elem>
                      <Elem name="field" mix="name">
                        {user.first_name} {user.last_name}
                      </Elem>
                      <Elem name="field" mix="last-activity">
                        {formatDistance(new Date(user.last_activity), new Date(), { addSuffix: true })}
                      </Elem>
                      <Elem name="field" mix="role" onClick={(event) => event.stopPropagation()}>
                        {canManageRoles ? (
                          <Select
                            value={role}
                            disabled={isOrganizationOwner}
                            options={[
                              { value: "AD", label: "Admin" },
                              { value: "MA", label: "Manager" },
                              { value: "ME", label: "Member" },
                            ]}
                            onChange={(event) => updateRole(membership, event.target.value)}
                          />
                        ) : (
                          role
                        )}
                      </Elem>
                    </Elem>
                  );
                })}
              </Elem>
            </Elem>
          ) : (
            <Elem name="loading">
              <Spinner size={36} />
            </Elem>
          )}
        </Elem>
        <Pagination
          page={currentPage}
          urlParamName="page"
          totalItems={totalItems}
          pageSize={currentPageSize}
          pageSizeOptions={[30, 50, 100]}
          onPageLoad={fetchUsers}
          style={{ paddingTop: 16 }}
        />
      </Block>
    </>
  );
};
