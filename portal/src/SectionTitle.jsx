import React from "react";
import { Group, Title, Text } from "@mantine/core";
export default function SectionTitle({ title, description, action }) {
  return (
    <Group justify="space-between" align="flex-end" mb={30}>
      <div>
        <Title order={1}>{title}</Title>
        {description && (
          <Text c="dimmed" mt={8}>
            {description}
          </Text>
        )}
      </div>
      {action}
    </Group>
  );
}
